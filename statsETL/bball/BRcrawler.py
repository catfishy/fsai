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
from collections import defaultdict
import traceback

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
    SEASONS = [str(x) for x in range(2005,2020)]

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
        recent_names = [_.strip() for _ in name_strings]
        data = {'location': loc_string,
                'name': recent_names,
                'url': url}
        # FIND ROSTER URL
        season_table = soup.find('table', id=team_id)
        roster_urls = [self.LINK_BASE + r.find('td').find('a')['href'] for r in season_table.find('tbody').findAll('tr')]
        lineup_urls = []

        # find valid rows
        valid_roster_urls = []
        for url in roster_urls:
            if any([cs in url for cs in self.SEASONS]):
                valid_roster_urls.append(url)

        if len(valid_roster_urls) == 0:
            self.logger.info("Not a current team")
            return {}

        # get lineup info for each year
        arenas = set()
        all_ids = set()
        for i, url in enumerate(valid_roster_urls):
            link_id = url.split('/')[-2].strip()
            if i == 0:
                data['_id'] = link_id
            all_ids.add(link_id)
            year = int(url.split('/')[-1].replace('.html',''))
            lineup_url = url.replace('.html','/lineups/')
           
            self.logger.info("Crawling INTO: %s (Year: %s)" % (url, year))
            content = self.getContent(url)
            r = re.compile(".*Arena:.*")
            entry = content.find_all(text=r)[0]
            arena = unicode(entry.parent.next_sibling.string)
            arena = arena.encode('ascii','ignore').replace(';','').strip()
            arenas.add(arena)

            #self.logger.info("Crawling INTO: %s (Year: %s)" % (lineup_url,year))
            #ln_content = self.getContent(lineup_url)
            # TODO: DO SOMETHING WITH LINE INFO WHEN USE CASE ARISES

        data['arenas'] = list(arenas)
        data['all_ids'] = list(all_ids)

        # merge with old data
        old_data = self.team_collection.find_one({"_id" : data['_id']})
        if old_data is None:
            old_data = {}
        old_data.update(data)

        self.logger.info("SAVING: %s" % old_data)
        self.nba_conn.saveDocument(self.team_collection, old_data)
        return old_data

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

        soup = self.checkVisit(url)

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
    BLACKLIST = ['pbp','shot-chart','plus-minus', 'baseball-reference', 'sports-reference', 
                 'hockey-reference', 'twitter', 'facebook', 'youtube', '/nbdl/']
    WHITELIST = ["/boxscores/"]
    PROCESSES = 3

    def __init__(self, refresh=False, logger=None, limit=None, days_back=1000, end_date=None):
        super(gameCrawler, self).__init__(logger=logger, limit=limit)
        self.name = "gameCrawler"
        # add tomorrow to blacklist to prevent going into the future
        # each box score page only has links to day before and after
        if end_date:
            self.custom_end = True
            self.end_date = end_date + timedelta(1)
        else:
            self.custom_end = False
            self.end_date = datetime.now() + timedelta(1)
        self.start_date = self.end_date - timedelta(int(days_back))

        self.BLACKLIST.append("index.cgi?month=%s&day=%s&year=%s" % (self.end_date.month,self.end_date.day,self.end_date.year))
        self.BLACKLIST.append("index.cgi?month=%s&day=%s&year=%s" % (self.start_date.month,self.start_date.day,self.start_date.year))

        self.upper_limit = self.end_date
        self.lower_limit = self.start_date

        print self.upper_limit
        print self.lower_limit

        self.refresh=refresh

        print self.BLACKLIST

        # get necessary collections
        self.nba_conn = MongoConn(db=NBA_DB)
        self.team_game_collection = self.nba_conn.getCollection("team_games")
        self.player_game_collection = self.nba_conn.getCollection("player_games")
        self.game_collection = self.nba_conn.getCollection("games")

        # create a player crawler in case need to crawl for players
        self.player_crawler = playerCrawler(refresh=True)
        self.player_crawler.createLogger()

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

    def run(self):
        '''
        run multiprocessed crawling
        '''
        # create logger
        if not self.logger:
            self.createLogger()
        self.logger.info("INIT: %s" % self.INIT_PAGE)
        self.logger.info("BLACKLIST: %s" % self.BLACKLIST)
        self.logger.info("WHITELIST: %s" % self.WHITELIST)
        # add initial page to queue
        if self.custom_end:
            new_init = "%sindex.cgi?month=%s&day=%s&year=%s" % (self.INIT_PAGE, self.end_date.month, self.end_date.day, self.end_date.year)
            self.queue.put(new_init, block=True)
            self.added.add(new_init)
        else:
            self.queue.put(self.INIT_PAGE,block=True)
            self.added.add(self.INIT_PAGE)
        self.mp_crawler = MultiprocessCrawler(type(self).__name__, self.logger, self.queue, self, num_processes=self.PROCESSES)
        self.mp_crawler.start()
        self.mp_crawler.join()


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

        try:
            team1_record = team1_insides[1].contents[0].split(' ')
            team2_record = team2_insides[1].contents[0].split(' ')
            team1_winloss = team1_record[0]
            team2_winloss = team2_record[0]
            team1_streak = re.sub('[\(\)]', '', ' '.join(team1_record[1:]), count=2)
            team2_streak = re.sub('[\(\)]', '', ' '.join(team2_record[1:]), count=2)
        except Exception as e:
            self.logger.info("PLAYOFF GAME, CONTINUING...")
            return {}

        gameinfo_table = final_insides[1].find('td').contents
        gametime = gameinfo_table[0]
        time_parts = [x.strip() for x in gametime.split(',')]
        if len(time_parts) == 2:
            # no time of day given, spoof 7 pm
            time_parts.insert(0,'7:00 PM')
        time_part_time = time_parts[0].split(' ')
        time_part_monthday = time_parts[1].split(' ')
        time_part_day = int(time_part_monthday[1])
        time_part_month = int(self.month_conv[time_part_monthday[0][:3]])
        time_part_year = int(time_parts[2])
        time_part_hour = int(time_part_time[0].split(':')[0])
        time_part_minute = int(time_part_time[0].split(':')[1])
        if time_part_time[1].lower() == 'pm' and time_part_hour != 12:
            time_part_hour += 12
        if time_part_time[1].lower() == 'am' and time_part_hour == 12:
            time_part_hour = 0

        gametime = datetime(year=time_part_year, month=time_part_month, day=time_part_day, hour=time_part_hour, minute=time_part_minute)

        try:
            gamelocation = gameinfo_table[1].string
        except Exception as e:
            gamelocation = None

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

        # pull more team stats from /boxscores/shot-chart/<id>.html
        shotcharturl = "%sshot-chart/%s.html" % (self.INIT_PAGE, game_id)
        # get content
        try:
            shotchart_soup = self.getContent(shotcharturl)
            self.logger.info("Crawling into %s" % shotcharturl)
        except Exception as e:
            self.logger.info("%s no content: %s" % (url,e))
            raise Exception("No shot chart content for %s" % url)

        team1_div_id = "shooting-%s" % team1_id.upper()
        team2_div_id = "shooting-%s" % team2_id.upper()
        team1_chart = shotchart_soup.find("table", id=team1_div_id)
        team2_chart = shotchart_soup.find("table", id=team2_div_id)

        team1_chart('colgroup')[0].extract()
        team1_chart.thead.unwrap()
        team1_chart.tbody.unwrap()
        team1_chart.tfoot.unwrap()
        team1_shooting_dict = [_ for _ in self.convert_html_table_to_dict(team1_chart) if _]
        team1_total = None
        for row in team1_shooting_dict:
            if row['Qtr'] == 'Tot':
                row.pop('Qtr', None)
                team1_total = row
                break

        team2_chart('colgroup')[0].extract()
        team2_chart.thead.unwrap()
        team2_chart.tbody.unwrap()
        team2_chart.tfoot.unwrap()
        team2_shooting_dict = [_ for _ in self.convert_html_table_to_dict(team2_chart) if _]
        team2_total = None
        for row in team2_shooting_dict:
            if row['Qtr'] == 'Tot':
                row.pop('Qtr', None)
                team2_total = row
                break

        # Save scoring_dict and factors_dict to respective team/game stat rows
        for scoring_stat in scoring_dict:
            # update with factors info
            for factor_stat in factors_dict:
                if factor_stat['team_id'] == scoring_stat['team_id']:
                    scoring_stat.update(factor_stat)
                    break
            # update with shooting stats
            if scoring_stat['team_id'] == team1_id:
                scoring_stat.update(team1_total)
            elif scoring_stat['team_id'] == team2_id:
                scoring_stat.update(team2_total)

            # update with location info
            if scoring_stat['team_id'] == data['away_id']:
                scoring_stat['location'] = "Away"
            else:
                scoring_stat['location'] = "Home"

        # Parse player stats
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
            starters = []
            for i, d in enumerate(v_dict):
                d['game_id'] = game_id
                d['game_time'] = data['time']
                # find the player id
                player_name = d['Starters']
                player_id_link = v.find('a', href=True, text=player_name)['href']
                player_id = player_id_link.split('/')[-1].replace('.html','')
                d['player_id'] = player_id

                if i < 5:
                    starters.append(player_id)

                d['player_team'] = table_team
                d.pop('Starters')
                # convert dnp to 0 mins
                if d['MP'] == "Did Not Play":
                    d['MP'] = 0.0
                elif d['MP'] == "Player Suspended":
                    d['MP'] = 0.0
                elif d['MP'] is None:
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

                # CRAWL PLAYER IF PLAYER NOT IN DATABASE
                if not player_collection.find_one({"_id" : d['player_id']}):
                    pid = d['player_id']
                    # recreate url
                    add_on = "%s/%s.html" % (pid[0], pid)
                    url = self.player_crawler.INIT_PAGE + add_on
                    # crawl player
                    self.logger.info("CRAWLING PLAYER: %s" % url)
                    self.player_crawler.crawlPage(url)

            if 'home_starters' not in data:
                if table_team == data['home_id']:
                    data['home_starters'] = starters
            if 'away_starters' not in data:
                if table_team.strip() == data['away_id']:
                    data['away_starters'] = starters

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

        # CALCULATE NORMALIZED BY 100 POSSESSION STATS for the player stats
        player_game_stats = self.calculateNormalizedStats(player_game_stats, scoring_dict)

        # ALL THE SAVING
        self.logger.info("Attempting SAVING: %s" % data)
        for scoring_stat in scoring_dict:
            try:
                self.nba_conn.saveDocument(self.team_game_collection, scoring_stat)
            except DuplicateKeyError as e:
                self.logger.info("Already saved team game...")
                continue
        for player_name, player_stats in player_game_stats.iteritems():
            try:
                self.nba_conn.saveDocument(self.player_game_collection, player_stats)
            except DuplicateKeyError as e:
                self.logger.info("Already saved player game...")
                continue
        try:
            self.nba_conn.saveDocument(self.game_collection, data)
        except DuplicateKeyError as e:
            self.logger.info("Already saved game...")

        return data

    def calculateNormalizedStats(self, player_game_stats, team_scoring_stats):
        '''
        Making the calculations according to http://www.basketball-reference.com/about/ratings.html
        '''
        # sum the scoring stats by team
        to_norm = ["FT", "3P", "TOV", "FG", "3PA", "DRB", "AST", "PF", 
                   "PTS", "FGA", "STL", "TRB", "FTA", "BLK", "ORB"]
        team_sum_keys = ['FGA', 'FT', 'FTA', 'AST', 'MP', 'PTS', 'ORB', 'TRB', 'DRB' ,'TOV', '3P', 'STL', 'BLK', 'PF', 'FG']
        team_stats = {}
        for row in team_scoring_stats:
            team_stats[row['team_id']] = {k : 0.0 for k in team_sum_keys}
        for player_name, player_row in player_game_stats.iteritems():
            if player_row['MP'] == 0.0:
                continue
            tid = player_row['player_team']
            for k in team_sum_keys:
                team_stats[tid][k] += float(player_row[k])
        both_teams = team_stats.keys()
        # calculate normalized by 100 possessions for each player row
        for player_name, player_row in player_game_stats.iteritems():
            if player_row['MP'] == 0.0:
                continue
            team = team_stats[player_row['player_team']]
            opp_team = both_teams[0] if both_teams[0] != player_row['player_team'] else both_teams[1]
            opponent = team_stats[opp_team]
            # basic
            PTS = float(player_row['PTS'])
            FGA = float(player_row['FGA'])
            FTM = float(player_row['FT'])
            FTA = float(player_row['FTA'])
            MP = float(player_row['MP'])
            TOV = float(player_row['TOV'])
            AST = float(player_row['AST'])
            STL = float(player_row['STL'])
            BLK = float(player_row['BLK'])
            DRB = float(player_row['DRB'])
            ORB = float(player_row['ORB'])
            FGM = float(player_row['FG'])
            PF = float(player_row['PF'])
            Three_PM = float(player_row['3P'])
            try:
                # offensive
                qAST = ((MP / (team['MP'] / 5)) * (1.14 * ((team['AST'] - AST) / team['FG']))) + ((((team['AST'] / team['MP']) * MP * 5 - AST) / ((team['FG'] / team['MP']) * MP * 5 - FGM)) * (1 - (MP / (team['MP'] / 5))))
                FG_Part = FGM * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * qAST) if FGA > 0.0 else 0.0
                AST_Part = 0.5 * (((team['PTS'] - team['FT']) - (PTS - FTM)) / (2 * (team['FGA'] - FGA))) * AST
                FT_Part = (1-(1-(FTM/FTA))**2)*0.4*FTA if FTA > 0.0 else 0.0
                Team_Scoring_Poss = team['FG'] + (1 - (1 - (team['FT'] / team['FTA']))**2) * team['FTA'] * 0.4
                Team_ORB_percent = team['ORB'] / (team['ORB'] + (opponent['TRB'] - opponent['ORB']))
                Team_Play_percent = Team_Scoring_Poss / (team['FGA'] + team['FTA'] * 0.4 + team['TOV'])
                Team_ORB_Weight = ((1 - Team_ORB_percent) * Team_Play_percent) / ((1 - Team_ORB_percent) * Team_Play_percent + Team_ORB_percent * (1 - Team_Play_percent))
                ORB_Part = ORB * Team_ORB_Weight * Team_Play_percent
                ScPoss = (FG_Part + AST_Part + FT_Part) * (1 - (team['ORB'] / Team_Scoring_Poss) * Team_ORB_Weight * Team_Play_percent) + ORB_Part
                FGxPoss = (FGA - FGM) * (1 - 1.07 * Team_ORB_percent)
                FTxPoss = ((1 - (FTM / FTA))**2) * 0.4 * FTA if FTA > 0.0 else 0.0
                TotPoss = ScPoss + FGxPoss + FTxPoss + TOV
                PProd_FG_Part = 2 * (FGM + 0.5 * Three_PM) * (1 - 0.5 * ((PTS - FTM) / (2 * FGA)) * qAST) if FGA > 0.0 else 0.0
                PProd_AST_Part = 2 * ((team['FG'] - FGM + 0.5 * (team['3P'] - Three_PM)) / (team['FG'] - FGM)) * 0.5 * (((team['PTS'] - team['FT']) - (PTS - FTM)) / (2 * (team['FGA'] - FGA))) * AST
                PProd_ORB_Part = ORB * Team_ORB_Weight * Team_Play_percent * (team['PTS'] / (team['FG'] + (1 - (1 - (team['FT'] / team['FTA']))**2) * 0.4 * team['FTA']))
                PProd = (PProd_FG_Part + PProd_AST_Part + FTM) * (1 - (team['ORB'] / Team_Scoring_Poss) * Team_ORB_Weight * Team_Play_percent) + PProd_ORB_Part
                Floor_percent = ScPoss / TotPoss if TotPoss > 0.0 else None
                # defensive
                Team_Possessions = opponent['FGA'] + (0.44 * opponent['FT']) + opponent['TOV'] - opponent['ORB']
                DOR_percent = opponent['ORB'] / (opponent['ORB'] + team['DRB'])
                DFG_percent = opponent['FG'] / opponent['FGA']
                FMwt = (DFG_percent * (1 - DOR_percent)) / (DFG_percent * (1 - DOR_percent) + (1 - DFG_percent) * DOR_percent)
                Stops1 = STL + BLK * FMwt * (1 - 1.07 * DOR_percent) + DRB * (1 - FMwt)
                Stops2 = (((opponent['FGA'] - opponent['FG'] - team['BLK']) / team['MP']) * FMwt * (1 - 1.07 * DOR_percent) + ((opponent['TOV'] - team['STL']) / team['MP'])) * MP + (PF / team['PF']) * 0.4 * opponent['FTA'] * (1 - (opponent['FT'] / opponent['FTA']))**2
                Stops = Stops1 + Stops2
                Stop_percent = (Stops * opponent['MP']) / (Team_Possessions * MP)
                # redundancies
                ORtg = 100 * (PProd / TotPoss) if TotPoss > 0.0 else None
                Team_Defensive_Rating = 100 * (opponent['PTS'] / Team_Possessions)
                D_Pts_per_ScPoss = opponent['PTS'] / (opponent['FG'] + (1 - (1 - (opponent['FT'] / opponent['FTA']))**2) * opponent['FTA']*0.4)
                DRtg = Team_Defensive_Rating + 0.2 * (100 * D_Pts_per_ScPoss * (1 - Stop_percent) - Team_Defensive_Rating)
            except Exception as e:
                print player_row
                traceback.print_exc()
                raise e
            # calculate possession normalized stats
            norm_stats = {'FloorPercent' : Floor_percent,
                          'TotPoss' : TotPoss,
                          'Stops' : Stops,
                          'StopPercent' : Stop_percent,
                          'PProd' : PProd}
            for k in to_norm:
                if TotPoss > 0.0:
                    normed = (float(player_row[k]) / TotPoss) * 100
                else:
                    normed = None
                norm_stats['%s_per100poss' % k] = normed
            # add calculations to row
            player_game_stats[player_name].update(norm_stats)

        return player_game_stats


    def itemExists(self, url):
        '''
        If the url is already in the db, skip it
        '''
        query = {"url": url}
        result = self.nba_conn.findOne(self.game_collection, query=query)
        return bool(result)

    def checkDateLimit(self, url):
        boxscore_regex = re.compile("/boxscores/[0-9]+")

        # filter irrelevant urls
        #"index.cgi?month=%s&day=%s&year=%s"
        if not (boxscore_regex.search(url) or ("index.cgi" in url and '/boxscores/' in url)):
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
        soup = self.checkVisit(url)

        if not soup:
            return False
        
        # extract links
        all_links = self.extract_links(soup)
        filtered_links = [_ for _ in all_links if self.checkDateLimit(_)]
        new_links = self.addLinksToQueue(filtered_links)
        self.logger.info("Adding %s links to queue, Queue size: %s" % (new_links,self.queue.qsize()))
        # decide relevance to crawl
        if self.isGamePage(url, soup):
            self.logger.info("Crawling %s" % url)
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
                    text = text.replace("guard/forward", 'sg and sf')
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
        self.nba_conn.saveDocument(player_collection, data)

        # try crawling advanced stats
        try:
            self.crawlAdvancedStats(player_id, soup)
        except Exception as e:
            self.logger.exception("Error crawling advanced stats for %s" % player_id)

        # try crawling on/off stats (urls that include /on-off/)
        try:
            self.crawlOnOffStats(player_id, soup)
        except Exception as e:
            self.logger.exception("Error crawling on-off stats for %s" % player_id)

        # try crawling lineup stats (urls that include /lineups/)
        try:
            self.crawlLineupStats(player_id, soup)
        except Exception as e:
            self.logger.exception("Error crawling lineup stats for %s" % player_id)

        return data

    def crawlOnOffStats(self, pid, soup):
        links = soup.find_all(href=re.compile("/on-off/"))
        to_save = []
        for l in links:
            href = l['href']
            # get year
            try:
                year = int([_ for _ in href.split('/') if _][-1])
            except ValueError as e:
                continue
            # get content
            full = self.LINK_BASE + href
            content = self.getContent(full)
            div = content.find(id="div_on-off")
            table = div.table
            # remove the first header row, and superfluous thead sections
            header_sections = table.find_all('thead')
            for extra_header in header_sections[1:]:
                extra_header.extract()
            header_rows = table.thead('tr')
            header_rows[0].extract()
            table.thead.unwrap()
            table.tbody.unwrap()
            onoff_stats = self.convert_html_table_to_dict(table, use_data_stat=True)
            # interested in rows where Split is "On - Off"
            for stat_row in onoff_stats:
                split_label = stat_row.pop('split_id', '')
                if 'On' in split_label and 'Off' in split_label:
                    # convert to floats and save it
                    for k,v in stat_row.iteritems():
                        if not isinstance(v, float) and v is not None:
                            v = str(v).replace('+','')
                            try:
                                if '%' in v:
                                    v = float(v.replace('%','').strip()) / 100.0
                                else:
                                    v = float(v)
                            except ValueError as e:
                                pass
                                #print "Couldn't convert %s: %s to float" % (k,v)
                            stat_row[k] = v
                    stat_row['year'] = year
                    stat_row['player_id'] = pid
                    to_save.append(stat_row)
        # save
        for save_row in to_save:
            # spoof end of season
            end_season = datetime(day=1,month=7,year=save_row['year'])
            save_row['time'] = end_season
            try:
                self.logger.info("SAVING ONOFF: %s" % (save_row))
                #print all_values
                self.nba_conn.saveDocument(onoff_collection, save_row)
            except DuplicateKeyError as e:
                continue

    def crawlLineupStats(self, pid, soup):
        links = soup.find_all(href=re.compile("/lineups/"))
        to_save = []
        for l in links:
            href = l['href']
            # get year
            try:
                year = int([_ for _ in href.split('/') if _][-1])
            except ValueError as e:
                continue
            # get content
            full = self.LINK_BASE + href
            content = self.getContent(full)
            div = content.find(id='div_lineups-2-man')
            table = div.table
            # remove first header row
            header_rows = table.thead('tr')
            header_rows[0].extract()
            table.thead.unwrap()
            table.tbody.unwrap()
            lineup_stats = self.convert_html_table_to_dict(table, use_data_stat=True, use_csk=True)
            for stat_row in lineup_stats:
                pids = stat_row.pop('lineup', '')
                if pids == 'Player Average':
                    pid_list = [pid]
                else:
                    pid_list = pids.split(':')
                # convert to floats and save it
                for k,v in stat_row.iteritems():
                    if not isinstance(v, float) and v is not None:
                        v = str(v).replace('+','')
                        try:
                            if '%' in v:
                                v = float(v.replace('%','').strip()) / 100.0
                            else:
                                v = float(v)
                        except ValueError as e:
                            pass
                            #print "Couldn't convert %s: %s to float" % (k,v)
                        stat_row[k] = v
                stat_row['year'] = year
                stat_row['players'] = pid_list
                to_save.append(stat_row)
        # save
        for save_row in to_save:
            # spoof end of season
            end_season = datetime(day=1,month=7,year=save_row['year'])
            save_row['time'] = end_season
            # sort players and rearrange into player_one and player_two keys
            players = save_row.pop("players",[])
            players = sorted(players)
            if len(players) == 1:
                save_row['player_one'] = players[0]
                save_row['player_two'] = None
            elif len(players) == 2:
                save_row['player_one'] = players[0]
                save_row['player_two'] = players[1]
            try:
                self.logger.info("SAVING two man combo: %s" % (save_row))
                #print all_values
                self.nba_conn.saveDocument(two_man_collection, save_row)
            except DuplicateKeyError as e:
                #self.logger.exception(e)
                continue

    def crawlAdvancedStats(self, pid, soup):
        by_year_team = defaultdict(list)
        advanced_div = soup.find("div", id="div_advanced")
        shooting_div = soup.find("div", id="div_shooting")
        pbp_div = soup.find("div", id="div_advanced_pbp")
        pop_keys = ['season', 'tm', 'pos', 'age', 'lg', 'g', 'lg_id', 'team_id', '']

        # scrape advanced
        advanced_table = advanced_div.table
        advanced_table.tfoot.extract()
        advanced_table.thead.unwrap()
        advanced_table.tbody.unwrap()
        advanced_stats = self.convert_html_table_to_dict(advanced_table)
        for row in advanced_stats:
            season = row['Season']
            team = row['Tm']
            if team.upper() == 'TOT':
                # means the season was split between teams, just save team rows
                continue
            begin_year = int(season.split('-')[0])
            key = (begin_year, team)
            filtered = {}
            for k,v in row.iteritems():
                if k.lower() in pop_keys:
                    continue
                if not isinstance(v, float) and v is not None:
                    try:
                        v = float(v)
                    except ValueError as e:
                        if len(v) == 0:
                            v = None
                        else:
                            pass
                            #print "Couldn't convert %s: %s to float" % (k,v)
                filtered[k] = v
            by_year_team[key].append(filtered)

        # scrape shooting
        shooting_table = shooting_div.table
        shooting_table.tfoot.extract()
        # remove the first two header rows 
        header_rows = shooting_table.thead('tr')
        header_rows[0].extract()
        header_rows[1].extract()
        shooting_table.thead.unwrap()
        shooting_table.tbody.unwrap()
        shooting_stats = self.convert_html_table_to_dict(shooting_table, use_data_stat=True)
        for row in shooting_stats:
            season = row['season']
            team = row['team_id']
            if team.upper() == 'TOT':
                # means the season was split between teams, just save team rows
                continue
            begin_year = int(season.split('-')[0])
            key = (begin_year, team)
            filtered = {}
            for k,v in row.iteritems():
                if k.lower() in pop_keys:
                    continue
                if not isinstance(v, float) and v is not None:
                    try:
                        v = float(v)
                    except ValueError as e:
                        if len(v) == 0:
                            v = None
                        else:
                            pass
                            #print "Couldn't convert %s: %s to float" % (k,v)
                filtered[k] = v
            by_year_team[key].append(filtered)

        # scrape pbp
        pbp_table = pbp_div.table
        pbp_table.tfoot.extract()
        # remove the first header row
        header_rows = pbp_table.thead('tr')
        header_rows[0].extract()
        pbp_table.thead.unwrap()
        pbp_table.tbody.unwrap()
        pbp_stats = self.convert_html_table_to_dict(pbp_table, use_data_stat=True)
        for row in pbp_stats:
            season = row['season']
            team = row['team_id']
            if team.upper() == 'TOT':
                # means the season was split between teams, just save team rows
                continue
            begin_year = int(season.split('-')[0])
            key = (begin_year, team)
            filtered = {}
            for k,v in row.iteritems():
                if k.lower() in pop_keys:
                    continue
                if not isinstance(v, float) and v is not None:
                    try:
                        if '%' in str(v):
                            v = float(str(v).replace('%','').strip()) / 100.0
                        else:
                            v = float(v)
                    except ValueError as e:
                        if len(v) == 0:
                            v = None
                        else:
                            pass
                            #print "Couldn't convert %s: %s to float" % (k,v)
                filtered[k] = v
            by_year_team[key].append(filtered)

        convert_keys = {'pct_1' : 'pct_pg',
                        'pct_2' : 'pct_sg',
                        'pct_3' : 'pct_sf',
                        'pct_4' : 'pct_pf',
                        'pct_5' : 'pct_c'}

        # save
        for k, values in by_year_team.items():
            #print "%s: %s" % (k, len(values))
            begin_year, team = k
            all_values = {}
            for v in values:
                all_values.update(v)
            # spoof end of season
            end_season = datetime(day=1,month=7,year=begin_year+1)
            all_values['player_id'] = pid
            all_values['team_id'] = team.upper()
            all_values['time'] = end_season
            for oldk, newk in convert_keys.iteritems():
                if oldk in all_values:
                    all_values[newk] = all_values[oldk] if all_values[oldk] is not None else 0.0
                    all_values.pop(oldk, None)
            try:
                self.logger.info("SAVING ADVANCED: %s, %s, %s" % (pid, team.upper(), end_season))
                #print all_values
                self.nba_conn.saveDocument(advanced_collection, all_values)
            except DuplicateKeyError as e:
                continue

    def itemExists(self, url):
        '''
        If the url is already in the db, skip it
        '''
        query = {"url": url}
        result = self.nba_conn.findOne(player_collection, query=query)
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
        soup = self.checkVisit(url)

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



