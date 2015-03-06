import sys
import multiprocessing as mp
import logging
import Queue
import re
import math
from datetime import datetime, timedelta
from xml.etree import ElementTree as ET
import calendar
import time

from bs4 import BeautifulSoup, element
import requests
from pymongo.helpers import DuplicateKeyError

from statsETL.db.mongolib import *
from statsETL.bball.NBAcrawler import crawlUpcomingGames
from statsETL.util.crawler import *
from analysis.util.kimono import updateNBARosters
from analysis.bball.gameAnalysis import modelPlayersInUpcomingGames


class teamCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/teams/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ['html']
    WHITELIST = ["/teams/"]
    PROCESSES = 4
    CURRENT_SEASON = "2015"

    def __init__(self, refresh=False, logger=None):
        super(teamCrawler, self).__init__(logger=logger)
        self.name = "teamCrawler"
        self.nba_conn = MongoConn(db=NBA_DB)
        self.team_collection = self.nba_conn.getCollection("teams")

        self.refresh=refresh

    def isTeamPage(self, url, soup):
        '''
        - check url fits pattern
        '''
        # check url
        url_pattern = re.compile("/teams/[A-Z]+/$")
        result = url_pattern.search(url)
        if not result:
            return False
        return True

    def crawlTeamPage(self, url, soup):
        team_id = url.split('/')[-2].strip()
        stw_boxes = soup("div", class_="stw")
        stw_box = stw_boxes[0]
        # get first p
        p_s = stw_box("p")[0]
        loc_span = p_s("span")[0]
        loc_string = loc_span.next_sibling.strip()
        name_span = p_s("span")[1]
        name_strings = name_span.next_sibling.split(',')
        recent_name = name_strings[0].strip()
        data = {'location': loc_string,
                'name': recent_name,
                'url': url}
        # FIND ROSTER URL
        season_table = soup.find('table', id=team_id)
        roster_url = season_table.find('tbody').find('tr').find('td').find('a')['href']
        roster_url = self.LINK_BASE + roster_url
        if self.CURRENT_SEASON not in roster_url:
            self.logger.info("Not a current team")
            return {}
        self.logger.info("Crawling INTO: %s" % roster_url)
        # load the id
        data['_id'] = roster_url.split('/')[-2].strip()
        roster_response = requests.get(roster_url, timeout=10)
        roster_content = roster_response.content
        roster_soup = BeautifulSoup(roster_content)
        roster_table = roster_soup.find("table", id="roster")
        roster_rows = roster_table.findAll('a', href=re.compile('players'))
        roster_ids = []
        for link in roster_rows:
            href = link['href']
            player_id = href.split('/')[-1].strip().replace('.html','')
            if player_id:
                roster_ids.append(player_id)

        #data['players'] = roster_ids

        self.logger.info("SAVING: %s" % data)
        self.nba_conn.saveDocument(self.team_collection, data)
        return data

    def itemExists(self, url):
        '''
        If the url is already in the db, skip it
        '''
        query = {"url": url}
        result = self.nba_conn.findOne(self.team_collection, query=query)
        return bool(result)

    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        # check if url already in db, if so, don't crawl again
        if not self.refresh and self.itemExists(url):
            self.logger.info("%s already crawled in past, skipping" % url)
            return False

        self.url_check_lock.acquire()
        soup = self.checkVisit(url)
        self.url_check_lock.release()

        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isTeamPage(url, soup):
            self.crawlTeamPage(url, soup)
            return True
        return False

class gameCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/boxscores/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ['pbp','shot-chart','plus-minus']
    WHITELIST = ["/boxscores/"]
    PROCESSES = 4

    def __init__(self, refresh=False, logger=None, limit=None, days_back=None):
        super(gameCrawler, self).__init__(logger=logger, limit=limit)
        self.name = "gameCrawler"
        # add tomorrow to blacklist to prevent going into the future
        # each box score page only has links to day before and after
        now = datetime.now() + timedelta(1) # one day from now
        day = now.day
        month = now.month
        year = now.year
        dateblacklist = "index.cgi?month=%s&day=%s&year=%s" % (month,day,year)
        self.BLACKLIST.append(dateblacklist)

        self.upper_limit = now
        self.lower_limit = None

        if days_back:
            now = datetime.now() - timedelta(int(days_back))
            day = now.day
            month = now.month
            year = now.year
            dateblacklist = "index.cgi?month=%s&day=%s&year=%s" % (month,day,year)
            self.BLACKLIST.append(dateblacklist)
            self.lower_limit = now

        print self.upper_limit
        print self.lower_limit

        self.refresh=refresh

        print self.BLACKLIST

        # get necessary collections
        self.nba_conn = MongoConn(db=NBA_DB)
        self.team_game_collection = self.nba_conn.getCollection("team_games")
        self.player_game_collection = self.nba_conn.getCollection("player_games")
        self.game_collection = self.nba_conn.getCollection("games")

    def isGamePage(self, url, soup):
        '''
        - check url fits pattern
        '''
        # check url
        url_pattern = re.compile("/boxscores/[0-9]+[A-Z]+\.html$")
        result = url_pattern.search(url)
        if not result:
            return False
        return True

    def crawlGamePage(self, url, soup):
        game_id = url.split('/')[-1].replace('.html','').strip().upper()
        page_content = soup.find('div', id="page_content").find('table')
        all_tables = page_content.find('tr').find('td')
        all_divs = all_tables('div',recursive=False)
        boxscore_div = all_divs[0]
        stats_div = all_divs[1]

        # parse boxscore
        float_left = boxscore_div.find('div', class_='float_left')
        box_tables = float_left('table', recursive=False)
        final_table = box_tables[1].find('tr').find('td').find('table')
        final_insides = final_table('tr', recursive=False)
        finalscore_table = final_insides[0]
        finalscore_table_insides = finalscore_table('td', recursive=False)
        team1_insides = finalscore_table_insides[0]('br')
        team2_insides = finalscore_table_insides[1]('br')
        team1_finalscore = team1_insides[0].string
        team2_finalscore = team2_insides[0].string
        team1_record = team1_insides[1].contents[0].split(' ')
        team2_record = team2_insides[1].contents[0].split(' ')
        team1_winloss = team1_record[0]
        team2_winloss = team2_record[0]
        team1_streak = re.sub('[\(\)]', '', ' '.join(team1_record[1:]), count=2)
        team2_streak = re.sub('[\(\)]', '', ' '.join(team2_record[1:]), count=2)

        gameinfo_table = final_insides[1].find('td').contents
        gametime = gameinfo_table[0]
        time_parts = [x.strip() for x in gametime.split(',')]
        time_part_time = time_parts[0].split(' ')
        time_part_monthday = time_parts[1].split(' ')
        time_part_day = int(time_part_monthday[1])
        time_part_month = int(self.month_conv[time_part_monthday[0][:3]])
        time_part_year = int(time_parts[2])
        time_part_hour = int(time_part_time[0].split(':')[0])
        time_part_minute = int(time_part_time[0].split(':')[1])
        if time_part_time[1].lower() == 'pm':
            time_part_hour += 12

        gametime = datetime(year=time_part_year, month=time_part_month, day=time_part_day, hour=time_part_hour, minute=time_part_minute)

        gamelocation = gameinfo_table[1].string
        team_id_regex = re.compile("/[A-Z]+/")
        team1_id = team_id_regex.search(finalscore_table_insides[0].find('a')['href']).group(0).replace('/','')
        team2_id = team_id_regex.search(finalscore_table_insides[1].find('a')['href']).group(0).replace('/','')

        data = {'_id': game_id,
                'away_id': team1_id,
                'home_id': team2_id,
                'teams': [team1_id, team2_id],
                'away_pts': float(team1_finalscore),
                'home_pts': float(team2_finalscore),
                'away_record': team1_winloss,
                'home_record': team2_winloss,
                'away_streak': team1_streak,
                'home_streak': team2_streak,
                'time': gametime,
                'location': gamelocation,
                'url': url}

        teamstat_table = box_tables[2].find('tr')
        teamstat_insides = teamstat_table('td', recursive=False)
        scoring_table = teamstat_insides[0].find('table')
        factors_table = teamstat_insides[1].find('table')
        # remove first row from scoring table
        scoring_table('tr')[0].extract()
        # remove colgroup, unwrap thead and tbody, then remove first row
        factors_table('colgroup')[0].extract()
        factors_table.thead.unwrap()
        factors_table.tbody.unwrap()
        factors_table('tr')[0].extract()

        scoring_dict = self.convert_html_table_to_dict(scoring_table)
        factors_dict = self.convert_html_table_to_dict(factors_table)
        
        # empty string key is the team id
        for d in scoring_dict:
            d['team_id'] = d['']
            d.pop('')
            d['game_id'] = game_id
            d['game_time'] = data['time']
        for d in factors_dict:
            d['team_id'] = d['']
            d.pop('')
            d['game_id'] = game_id

        # Save scoring_dict and factors_dict to respective team/game stat rows
        for scoring_stat in scoring_dict:
            # update with factors info
            for factor_stat in factors_dict:
                if factor_stat['team_id'] == scoring_stat['team_id']:
                    scoring_stat.update(factor_stat)
                    break
            if scoring_stat['team_id'] == data['away_id']:
                scoring_stat['location'] = "Away"
            else:
                scoring_stat['location'] = "Home"

        # parse player stats
        stats_tables = stats_div('div', class_='table_container')
        stats_tables_by_id = {tablediv['id'].replace('div_','') : tablediv.table for tablediv in stats_tables}
        player_game_stats = {}
        for k,v in stats_tables_by_id.iteritems():
            table_team = k.split('_')[0]
            # remove colgroup
            v('colgroup')[0].extract()
            v.thead.unwrap()
            v.tbody.unwrap()
            v('tr')[0].extract()
            v_dict = [_ for _ in self.convert_html_table_to_dict(v) if _]
            # player name in column u'Starters' 
            for d in v_dict:
                d['game_id'] = game_id
                d['game_time'] = data['time']
                # find the player id
                player_name = d['Starters']
                player_id_link = v.find('a', href=True, text=player_name)['href']
                player_id = player_id_link.split('/')[-1].replace('.html','')
                d['player_id'] = player_id
                d['player_team'] = table_team
                d.pop('Starters')
                # convert dnp to 0 mins
                if d['MP'] == "Did Not Play":
                    d['MP'] = 0.0
                elif d['MP'] == "Player Suspended":
                    d['MP'] = 0.0
                else:
                    minutessecs = d['MP'].split(':')
                    minutes = float(minutessecs[0])
                    seconds = float(minutessecs[1])
                    seconds_frac = seconds/60.0
                    d['MP'] = minutes + seconds_frac
                # add to player row
                if d['player_id'] in player_game_stats:
                    player_game_stats[d['player_id']].update(d)
                else:
                    player_game_stats[d['player_id']] = d

        # parse game info at bottom of page
        lower_gameinfo_table = stats_div('table', recursive=False)[0]
        for row in lower_gameinfo_table('tr', recursive=False):
            title_name = row('td')[0].string.lower().replace(':','')
            if title_name == 'attendance':
                data['attendance'] = int(row('td')[1].string.replace(',',''))
            elif title_name == 'time of game':
                data['time of game'] = row('td')[1].string
            elif title_name == 'officials':
                official_links = row('td')[1]('a', recursive=False)
                official_ids = [l['href'].split('/')[-1].replace('.html','') for l in official_links]
                data['officials'] = official_ids
            elif title_name == 'inactive':
                inactive_links = row('td')[1]('a', recursive=False)
                inactive_ids = [l['href'].split('/')[-1].replace('.html','') for l in inactive_links]
                data['inactive'] = inactive_ids

        # ALL THE SAVING
        for scoring_stat in scoring_dict:
            try:
                self.nba_conn.saveDocument(self.team_game_collection, scoring_stat)
            except DuplicateKeyError as e:
                continue
        for player_name, player_stats in player_game_stats.iteritems():
            try:
                self.nba_conn.saveDocument(self.player_game_collection, player_stats)
            except DuplicateKeyError as e:
                continue
        self.logger.info("SAVING: %s" % data)
        self.nba_conn.saveDocument(self.game_collection, data)
        return data

    def itemExists(self, url):
        '''
        If the url is already in the db, skip it
        '''
        query = {"url": url}
        result = self.nba_conn.findOne(self.game_collection, query=query)
        return bool(result)

    def checkDateLimit(self, url):
        boxscore_regex = re.compile("/boxscores/[0-9]+")
        #"index.cgi?month=%s&day=%s&year=%s"
        # filter irrelevant urls
        if "index.cgi" not in url and '/boxscores/' not in url:
            return True
        if not boxscore_regex.search(url):
            return True

        if "index.cgi" in url:
            url_date = url.split('?')[-1]
            url_date_parts = url_date.split('&')
            month = int(url_date_parts[0].replace('month=',''))
            day = int(url_date_parts[1].replace('day=',''))
            year = int(url_date_parts[2].replace('year=',''))
            date = datetime(year=year,month=month,day=day)
        elif '/boxscores/' in url:
            gameid = url.split('/boxscores/')[-1]
            if '.html' not in gameid:
                return True
            gameid = gameid.replace('.html','').strip().upper()
            # get first 8 chars, represents the date
            firsteight = gameid[:8]
            year = int(firsteight[:4])
            month = int(firsteight[4:6])
            day = int(firsteight[6:])
            date = datetime(year=year,month=month,day=day)
        if date > self.upper_limit:
            return False
        elif self.lower_limit is not None and date <= self.lower_limit:
            return False
        return True

    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        # check if url already in db, if so, don't crawl again
        if not self.refresh and self.itemExists(url):
            self.logger.info("%s already crawled in past, skipping" % url)
            return False
        time.sleep(0.1)
        self.url_check_lock.acquire()
        soup = self.checkVisit(url)
        self.url_check_lock.release()

        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        filtered_links = [_ for _ in all_links if self.checkDateLimit(_)]
        new_links = self.addLinksToQueue(filtered_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isGamePage(url, soup):
            self.crawlGamePage(url, soup)
            return True
        return False


class playerCrawler(Crawler):

    INIT_PAGE = "http://www.basketball-reference.com/players/"
    LINK_BASE = "http://www.basketball-reference.com"
    BLACKLIST = ["gamelog", "splits", "news.cgi","shooting", "lineups", "on-off", "cbb", "http", "nbdl", "euro", "nbl"]
    WHITELIST = ["/players/"]
    PROCESSES = 8

    def __init__(self, refresh=False, logger=None):
        super(playerCrawler, self).__init__(logger=logger)
        self.name = "playerCrawler"
        self.nba_conn = MongoConn(db=NBA_DB)
        self.player_collection = self.nba_conn.getCollection("players")

        self.refresh=refresh

    def isPlayerPage(self, url, soup):
        '''
        - check url fits pattern
        - look for experience in info_box (for active player)
        '''
        # check url
        url_pattern = re.compile("/players/[a-z]/[a-z]+[0-9]+\.html")
        result = url_pattern.search(url)
        if not result:
            return False
        # check info_box
        info_boxes = soup(id="info_box")
        if len(info_boxes) > 0:
            info_box = info_boxes[0]
            spans = info_box(text=re.compile("Experience\:"))
            if len(spans) > 0:
                return True
        return False

    def crawlPlayerPage(self, url, soup):
        player_id = url.split('/')[-1].replace('.html','')
        self.logger.info("Crawling PLAYER: %s" % player_id)
        info_boxes = soup(id="info_box")
        info_box = info_boxes[0]
        name_p = info_box("p", class_="margin_top")[0]
        nickname = info_box('h1')[0].string
        full_name = name_p("span", class_="bold_text")[0].string
        stat_p = info_box("p", class_="padding_bottom_half")[0]
        stat_titles = stat_p("span", class_="bold_text")
        data = {'_id': player_id,
                'full_name': full_name,
                'nickname': nickname,
                'url': url
                }
        valid_fields = ['position','shoots','height','weight','born','nba debut','experience']
        for title_tag in stat_titles:
            title = title_tag.string
            title = title.replace(':','').lower().strip()
            if title in valid_fields:
                if title == 'born':
                    birth_span = soup(id="necro-birth")[0]
                    text = birth_span['data-birth']
                    ymd = [int(x.strip()) for x in text.split('-')]
                    text = datetime(year=ymd[0],month=ymd[1],day=ymd[2])
                elif title == 'nba debut':  
                    date_link = title_tag.next_sibling.next_sibling
                    date = date_link.string
                    date_components = date.split(' ')
                    month_num = int(self.month_conv[date_components[0][:3]])
                    date_num = int(date_components[1].replace(',',''))
                    year_num = int(date_components[2])
                    text = datetime(year=year_num,month=month_num,day=date_num)
                elif title == 'experience':
                    text = title_tag.next_sibling.strip().lower()
                    if text == 'rookie':
                        text = '0'
                    else:
                        text = text.replace('years','').strip()
                        text = text.replace('year','').strip()
                        text = int(text)
                elif title == 'weight':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace('lbs.','').strip()
                    text = int(text)
                elif title == 'height':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace(u"\xa0\u25aa", u' ').strip()
                    ft_inches = [int(x.strip()) for x in text.split('-')]
                    text = 12*ft_inches[0] + ft_inches[1]
                elif title == 'position':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace(u"\xa0\u25aa", u' ').strip()
                    text = text.replace('small forward','sf')
                    text = text.replace('power forward','pf')
                    text = text.replace('shooting guard','sg')
                    text = text.replace('point guard','pg')
                    text = text.replace('center','c')
                    if 'and' in text:
                        text = [x.strip() for x in text.split('and') if x.strip()]
                    else:
                        text = [text]
                elif title == 'shoots':
                    text = title_tag.next_sibling.strip().lower()
                    text = text.replace('left','L')
                    text = text.replace('right','R')               
                else:
                    text = title_tag.next_sibling.strip().lower()
                data[title] = text
        self.logger.info("SAVING: %s" % data)
        self.nba_conn.saveDocument(self.player_collection, data)
        return data

    def itemExists(self, url):
        '''
        If the url is already in the db, skip it
        '''
        query = {"url": url}
        result = self.nba_conn.findOne(self.player_collection, query=query)
        return bool(result)

    def crawlPage(self, url):
        '''
        Crawl a page
        '''
        # check if url already in db, if so, don't crawl again
        if not self.refresh and self.itemExists(url):
            self.logger.info("%s already crawled in past, skipping" % url)
            return False
        time.sleep(0.1)

        self.url_check_lock.acquire()
        soup = self.checkVisit(url)
        self.url_check_lock.release()

        if not soup:
            return False
        self.logger.info("Crawling %s" % url)
        # extract links
        all_links = self.extract_links(soup)
        new_links = self.addLinksToQueue(all_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isPlayerPage(url, soup):
            self.crawlPlayerPage(url, soup)
            return True
        return False


if __name__=="__main__":
    p_crawl = playerCrawler(refresh=True)
    t_crawl = teamCrawler(refresh=True)
    g_crawl = gameCrawler(refresh=True, days_back=7)

    p_crawl.run()
    g_crawl.run()
    t_crawl.run()



