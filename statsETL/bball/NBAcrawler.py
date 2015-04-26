from datetime import datetime, timedelta
from collections import defaultdict
import re
import csv
import StringIO

import requests
from bs4 import BeautifulSoup, element
from pymongo.helpers import DuplicateKeyError

from statsETL.db.mongolib import *
from statsETL.util.crawler import *

def crawlUpcomingGames(days_ahead=7):
    dates = []
    new_games = []
    for i in range(days_ahead):
        d = datetime.now() + timedelta(i)
        gl_crawl = upcomingGameCrawler(date=d)
        new_games = gl_crawl.crawlPage()
    return new_games

def updateNBARosterURLS(roster_urls):
    rcrawl = rosterCrawler(roster_urls)
    rcrawl.run()

def saveNBADepthChart():
    dcrawl = depthCrawler()
    dcrawl.run()

def crawlNBATrackingStats():
    pullup_address = ("pullup", "http://stats.nba.com/js/data/sportvu/pullUpShootData.js")
    drives_address = ("drives", "http://stats.nba.com/js/data/sportvu/drivesData.js")
    defense_address = ("defense", "http://stats.nba.com/js/data/sportvu/defenseData.js")
    passing_address = ("passing", "http://stats.nba.com/js/data/sportvu/passingData.js")
    touches_address = ("touches", "http://stats.nba.com/js/data/sportvu/touchesData.js")
    speed_address = ("speed", "http://stats.nba.com/js/data/sportvu/speedData.js")
    rebounding_address = ("rebounding", "http://stats.nba.com/js/data/sportvu/reboundingData.js")
    catchshoot_address = ("catchshoot", "http://stats.nba.com/js/data/sportvu/catchShootData.js")
    shooting_address = ("shooting", "http://stats.nba.com/js/data/sportvu/shootingData.js")
    addresses = [pullup_address, drives_address, defense_address, passing_address, 
                 touches_address, speed_address, rebounding_address, catchshoot_address,
                 shooting_address]
    player_stats = defaultdict(dict)
    for name, url in addresses:
        response = requests.get(url, timeout=10)
        content = response.content
        content = re.sub("[\{\}]","", content)
        content = re.sub("[\[\]]", "\n", content)
        content = re.sub("\"rowSet\":\n", "", content)
        content = re.sub(";", ",", content)
        rows = content.split('\n')
        rows = rows[2:]
        reader = csv.DictReader(rows)
        for row in reader:
            # find player id
            playername = row['PLAYER']
            if playername:
                # pop keys
                row.pop('FIRST_NAME', None)
                row.pop('LAST_NAME', None)
                row.pop('PLAYER', None)
                row.pop('PLAYER_ID', None)
                # change 'null' to None, or convert to float
                for k,v in row.iteritems():
                    try:
                        row[k] = float(v)
                    except Exception as e:
                        row[k] = None
                player_stats[playername].update(row)

    # match for player ids
    pids = translatePlayerNames(player_stats.keys())
    for name, prow in pids.iteritems():
        player_stats[name]['player_id'] = prow['_id']

    # save
    x = datetime.now()
    today = datetime(day=x.day,month=x.month,year=x.year)
    for playername, to_save in player_stats.iteritems():
        to_save['time'] = today
        try:
            nba_conn.saveDocument(espn_player_stat_collection, to_save)
        except DuplicateKeyError as e:
            print "%s espn player stats already saved" % today

def translatePlayerNames(player_list):
    player_dicts = {}
    unmatched = []
    for d in player_list:
        query = createPlayerRegex(d)
        full_matched = player_collection.find_one({'full_name' : {'$regex' : query}})
        nick_matched = player_collection.find_one({'nickname' : {'$regex' : query}})
        if nick_matched or full_matched:
            match_obj = nick_matched or full_matched # value nickname matches more
            player_dicts[d] = match_obj
        else:
            print "%s, no match: %s" % (d, query)
            unmatched.append(d)
    return player_dicts

def createPlayerRegex(raw_string):
    '''
    Includes Hardcoded translations
    '''
    translations = {"Patty": "Patrick",
                    "JJ": "J.J.",
                    "KJ": "K.J.",
                    "CJ": "C.J.",
                    "TJ": "T.J.",
                    "PJ": "P.J."}
    query = ''
    name_parts = [x.strip() for x in raw_string.split(' ') if x.strip()]
    for p in name_parts:
        if p in translations:
            p = translations[p]
        query += '.*%s' % p.replace('.','.*')
    if query:
        query += '.*'
    return re.compile(query)

def translateTeamNames(team_list):
    team_dicts = {}
    for t in team_list:
        query = createTeamRegex(t)
        matched = list(team_collection.find({'name' : {'$regex' : query}}))
        if len(matched) == 1:
            team_dicts[t] = matched[0]["_id"]
        elif len(matched) > 1:
            matchrank = defaultdict(list)
            # choose match with the most recent name
            for m in matched:
                m_names = m['name']
                for i, name in enumerate(m_names):
                    if re.search(query, name):
                        matchrank[i].append(m['_id'])
                        break
            best_rank_key = sorted(matchrank.keys())[0]
            best_rank_matches = matchrank[best_rank_key]
            if len(best_rank_matches) == 1:
                team_dicts[t] = best_rank_matches[0]
            else:
                print "%s, ambiguous match: %s" % (t, best_rank_matches)
        else:
            print "%s, no match %s" % (t, query)
    return team_dicts

def createTeamRegex(raw_string):
    '''
    Includes hardcoded translatiions
    '''
    translations = {"LA": "Los Angeles",
                    "NO/Oklahoma": "New Orleans/Oklahoma"}

    name_parts = [x.strip() for x in raw_string.split(' ') if x.strip()]
    query = ''
    for p in name_parts:
        if p in translations:
            p = translations[p]
        query += '.*%s' % p.replace('.','.*')
    if query:
        query += '.*'
    return re.compile(query)


class depthCrawler(Crawler):

    INIT_PAGE = "http://espn.go.com/nba/depth/_/type/print"

    def __init__(self, logger=None):
        super(depthCrawler, self).__init__(logger=logger)
        self.name = "depthCrawler"

    def run(self):
        if not self.logger:
            self.createLogger()
        self.logger.info("URL: %s" % self.INIT_PAGE)
        try:
            soup = self.getContent(self.INIT_PAGE)
        except Exception as e:
            self.logger.exception("%s no content: %s" % (self.INIT_PAGE,e))
            return False

        date = soup.find("font", class_="date").text.strip()
        date_obj = datetime.strptime("April 13, 2015", "%B %d, %Y")

        to_save = {"time" : date_obj,
                   "invalids": [],
                   "stats": {}}

        tables = soup.find_all("table")
        depth_table = tables[1]
        depth_cols = depth_table.find_all("td", class_="verb10")
        depth = {}

        # get all the team names and get the translations
        raw_teams = []
        for d in depth_cols:
            contents = d.font.contents
            team_name = contents[0].text.strip()
            raw_teams.append(team_name)
        teamnames = translateTeamNames(raw_teams)

        # parse the depth charts
        for d in depth_cols:
            contents = d.font.contents
            team_name = contents[0].text.strip()
            team_id = teamnames[team_name]

            # get the team roster
            team_roster = team_collection.find_one({"_id" : team_id})['players']
            team_roster_dict = {_ : player_collection.find_one({"_id" : _}) for _ in team_roster}

            # parse depth
            team_depth = [_.strip() for _ in contents[1].text.split('\n') if _]
            team_results = defaultdict(list)
            for raw_string in team_depth:
                parts = raw_string.split('-')
                pos = parts[0][:-1].lower()
                pos_rank = int(parts[0][-1])
                name = parts[1]

                # check injury
                injured = False
                if '(IL)' in name:
                    name = name.replace('(IL)','').strip()
                    injured = True

                # translate the name
                query = createPlayerRegex(name)
                translated_pid = None
                for pid, pid_row in team_roster_dict.iteritems():
                    # check if regex matches full name or nickname
                    if query.search(pid_row['full_name']) or query.search(pid_row['nickname']):
                        translated_pid = pid
                        break
                if not translated_pid:
                    self.logger.info("Could not match %s with %s roster" % (name, team_id))
                    continue

                team_results[pos].append((translated_pid, pos_rank))

                if injured:
                    to_save['invalids'].append(translated_pid)

            depth[team_id] = {}
            for k,v in team_results.items():
                sorted_pos = sorted(v, key=lambda x: x[1])
                depth[team_id][k] = [_[0] for _ in sorted_pos]

        to_save['stats'] = depth

        # save
        try:
            nba_conn.saveDocument(espn_depth_collection, to_save)
        except DuplicateKeyError as e:
            print "espn depth chart already saved for the day"
        
        return to_save


class rosterCrawler(Crawler):

    INIT_PAGE = ""

    def __init__(self, urls, logger=None):
        super(rosterCrawler, self).__init__(logger=logger)
        self.name = "rosterCrawler"
        self.urls = urls

    def run(self):
        if not self.logger:
            self.createLogger()
        self.logger.info("URLS: %s" % self.urls)
        for name, url in self.urls.iteritems():
            team_row = team_collection.find_one({"name":str(name).strip()})
            if not team_row:
                raise Exception("Could not find %s" % name)
            try:
                soup = self.getContent(url)
            except Exception as e:
                self.logger.exception("%s no content: %s" % (url,e))
                continue
            odds = soup('tr', class_="oddrow")
            evens = soup('tr', class_="evenrow")
            pnames = []
            for row in odds+evens:
                pnames.append(row('a')[0].string)
            player_dicts = translatePlayerNames(pnames)
            player_ids = [v["_id"] for k,v in player_dicts.iteritems()]
            team_row['players'] = player_ids
            team_collection.save(team_row)
            self.logger.info("Saved %s roster: %s" % (name, player_ids))

class upcomingGameCrawler(Crawler):

    INIT_PAGE = "http://www.nba.com/gameline/"

    def __init__(self, date=None, logger=None):
        super(upcomingGameCrawler, self).__init__(logger=logger)
        self.name = "gamelineCrawler"
        if not date:
            raise Exception("Must Specify a datetime")
        self.date = date # must be datetime object


    def crawlPage(self):
        if not self.logger:
            self.createLogger()
        # create url to crawl
        year = str(self.date.year)
        month = str(self.date.month).zfill(2)
        day = str(self.date.day).zfill(2)
        url_to_crawl = "%s%s%s%s" % (self.INIT_PAGE, year, month, day)
        print url_to_crawl

        # try to get the soup
        soup = self.checkVisit(url_to_crawl)

        if not soup:
            self.logger.info("could not find url %s" % url_to_crawl)
            return False

        return self.crawlGamelinePage(url_to_crawl, soup)

    def crawlGamelinePage(self, url, soup):
        recap_listings = soup('div', class_='Recap GameLine')
        live_listings = soup('div', class_='Live GameLine')
        preview_listings = soup('div', class_='Pre GameLine')
        
        preview_data = []

        # for preview boxes
        for preview in preview_listings:
            prescore_div = preview.find('div', class_="nbaPreMnScore")
            time_div = prescore_div.find('div', class_="nbaPreMnStatus")
            teams_div = prescore_div.find('div', class_="nbaPreMnTeamInfo")

            time_parts = [x for x in time_div.stripped_strings]
            teams_images = teams_div('img', title=True)
            teams_parts = [x['title'] for x in teams_images]

            gametime = self.parseTime(time_parts)
            data = {'time': gametime,
                    'home_team_name': teams_parts[1],
                    'away_team_name': teams_parts[0]
                    }
            preview_data.append(data)

        # for recap boxes
        for recap in recap_listings:
            score_div = recap.find('div', class_="nbaModTopScore")
            time_div = score_div.find('div', class_="nbaFnlStatTxSm")
            away_team_div = score_div.find('div', class_="nbaModTopTeamAw")
            home_team_div = score_div.find('div', class_="nbaModTopTeamHm")
            away_image = away_team_div.find('img', title=True)
            home_image = home_team_div.find('img', title=True)
            away_name = away_image['title']
            home_name = home_image['title']

            time_parts = [x for x in time_div.stripped_strings]
            time_parts = time_parts[0].split(' ')
            gametime = self.parseTime(time_parts)
            data = {'time': gametime,
                    'home_team_name': home_name,
                    'away_team_name': away_name
                    }
            preview_data.append(data)

        # for live games
        for live in live_listings:
            pass

        # save into future games db
        for d in preview_data:
            home_row = team_collection.find_one({"name":str(d['home_team_name']).strip()})
            away_row = team_collection.find_one({"name":str(d['away_team_name']).strip()})
            home_id = home_row['_id']
            away_id = away_row['_id']
            teams = [home_id, away_id]
            new_id = "%s@%s" % (away_id, home_id)
            d['_id'] = new_id
            d['teams'] = teams
            d['home_id'] = home_id
            d['away_id'] = away_id
            nba_conn.saveDocument(future_collection, d)

        return preview_data

    def parseTime(self, time_parts):
        hourminute = time_parts[0].split(':')
        ampm = time_parts[1]
        hour = int(hourminute[0])
        if 'pm' in ampm and hour != 12:
            hour += 12
        if 'am' in ampm and hour == 12:
            hour = 0
        minute = int(hourminute[1])
        gametime = datetime(year=self.date.year, month=self.date.month, day=self.date.day, hour=hour, minute=minute)
        return gametime

if __name__=="__main__":
    '''
    today = datetime.now() + timedelta(int(1))
    gl_crawl = upcomingGameCrawler(date=[today])
    gl_crawl.crawlPage()
    crawlNBATrackingStats()
    '''
    saveNBADepthChart()

