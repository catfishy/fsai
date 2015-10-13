from datetime import datetime, timedelta
from collections import defaultdict
import re
import csv
import StringIO
import requests
import pandas as pd
from bs4 import BeautifulSoup, element
from pymongo.helpers import DuplicateKeyError
import dateutil.parser
import traceback
import multiprocessing as mp
import sys

from statsETL.db.mongolib import *
from statsETL.util.crawler import *
from statsETL.bball.BRcrawler import crawlBRPlayer

years_back=[2009,2010,2011,2012,2013,2014,2015]


def crawlBRPlayerPosition(pid, player_name):
    player_name_parts = [_.lower().replace('.','').strip() for _ in player_name.split(' ')]
    if player_name_parts[-1] == 'jr':
        player_name_parts = player_name_parts[:-1]
    player_search = '+'.join(player_name_parts)
    url = "http://www.basketball-reference.com/search/search.fcgi?search=%s" % player_search
    print url
    links = getAllLinks(url)

    url_pattern = re.compile("/players/[a-z]/[a-z]+[0-9]+\.html")
    results = []
    valid_links = False
    for l in links:
        if not isinstance(l, str):
            url = "http://www.basketball-reference.com" + l['href']
        else:
            url = l
        result = url_pattern.search(url)
        if result:
            valid_links = True
            # crawl for position
            print url
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.content)
                possible_tags = soup.find_all("span", class_='bold_text')
                for pt in possible_tags:
                    if not pt.string:
                        continue
                    text = pt.string.strip().lower().replace(':','')
                    if text == 'position':
                        text = pt.next_sibling.strip().lower()
                        text = text.replace(u"\xa0\u25aa", u' ').strip()
                        text = text.replace("c/forward", 'c and pf')
                        text = text.replace("c/sf", 'c and sf')
                        text = text.replace("pf/guard", "pf and sg and pg")
                        text = text.replace("guard/forward", 'sg and sf')
                        text = text.replace('small forward','sf')
                        text = text.replace('power forward','pf')
                        text = text.replace('forward', 'sf and pf')
                        text = text.replace('shooting guard','sg')
                        text = text.replace('point guard','pg')
                        text = text.replace('center','c')
                        if 'and' in text:
                            text = [x.strip().upper() for x in text.split('and') if x.strip()]
                        else:
                            text = [text.upper()]
                        pos = text
                    elif text == 'born':
                        birth_span = soup(id="necro-birth")[0]
                        text = birth_span['data-birth']
                        ymd = [int(x.strip()) for x in text.split('-')]
                        birth = datetime(year=ymd[0],month=ymd[1],day=ymd[2])
                if birth and pos:
                    results.append((birth, pos))
    if not valid_links:
        print "Unable to crawl BR Player %s: %s, no valid links" % (pid, player_name)
    elif not results:
        print "Unable to crawl BR Player %s: %s" % (pid, player_name)
    return results

def crawlPlayerRotowire(pid):
    # TODO: WHAT DOES THE NUMBER IN THE URL SIGNIFY? 
    url = "http://stats.nba.com/feeds/RotoWirePlayers-583598/%s.json"
    wires = []
    try:
        response = requests.get(url)
        response = response.json()
        print response
        wires = sorted(response['PlayerRotowires'], key=lambda x: int(x['Date']), reverse=True)
    except Exception as e:
        print "Player Rotowire Exception: %s" % e
    return wires


def crawlUpcomingGames(days_ahead=7):
    new_games = []
    for i in range(days_ahead):
        date = datetime.now() + timedelta(i)
        # find games for the day
        url = "http://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate=%s" % date.strftime("%m/%d/%Y")
        results = turnJSONtoPD(url)
        games = results.get('GameHeader', None)
        if games is None or len(games) == 0:
            print "No games on %s" % date
            continue
        for i, g in games.iterrows():
            gid = g['GAME_ID']
            season = g['SEASON']
            home_id = g['HOME_TEAM_ID']
            visitor_id = g['VISITOR_TEAM_ID']
            summary = {}
            summary['HOME_TEAM_ID'] = home_id
            summary['VISITOR_TEAM_ID'] = visitor_id
            inactives = []
            players = []
            depth = {}
            # TODO: get depth chart
            # TODO: get roster
            data = {'_id': gid,
                    'season': season,
                    'date': date,
                    'players': players,
                    'depth': depth,
                    'InactivePlayers': inactives,
                    'teams': sorted([home_id, visitor_id]),
                    'GameSummary': df.DataFrame(summary)}
            try:
                nba_conn.saveDocument(nba_games_collection, data)
            except DuplicateKeyError as e:
                pass
            new_games.append(gid)
    return new_games

def updateNBARosterURLS(roster_urls):
    rcrawl = rosterCrawler(roster_urls)
    rcrawl.run()

def saveNBADepthChart():
    dcrawl = depthCrawler()
    dcrawl.run()

def crawlNBAGames(date_start,date_end):
    date_start = datetime.strptime(date_start, '%m/%d/%Y')
    date_end = datetime.strptime(date_end, "%m/%d/%Y")
    pool = mp.Pool(processes=12)
    arg_chunk = []
    def args(date_start, date_end):
        for date in daterange(date_start, date_end):
            print date
            # find games for the day
            url = "http://stats.nba.com/stats/scoreboardV2?DayOffset=0&LeagueID=00&gameDate=%s" % date.strftime("%m/%d/%Y")
            try:
                results = turnJSONtoPD(url)
            except Exception as e:
                print "Exception crawling game day page %s: %s" % (date, e)
                continue
            games = results.get('GameHeader', None)
            if games is None or len(games) == 0:
                print "No games on %s" % date
                continue
            for i, g in games.iterrows():
                gid = g['GAME_ID']
                season = g['SEASON']
                to_yield = (gid, date, season)
                yield to_yield
    print date_start
    print date_end
    for i, _ in enumerate(pool.imap_unordered(crawlNBAGameData, args(date_start, date_end)), 1): 
        pass
    pool.close()
    pool.join()

def updateGameData(url, data):
    print url
    newdata = turnJSONtoPD(url)
    for k,v in newdata.iteritems():
        if k not in data or data[k].empty:
            data[k] = v
        else:
            # merge
            if 'PLAYER_ID' in v:
                cols_to_use = (v.columns-data[k].columns).tolist()
                cols_to_use.append('PLAYER_ID')
                data[k] = pd.merge(data[k], v[cols_to_use], on='PLAYER_ID')
                print "merged into %s" % k
            elif 'TEAM_ID' in v:
                cols_to_use = (v.columns-data[k].columns).tolist()
                cols_to_use.append('TEAM_ID')
                data[k] = pd.merge(data[k], v[cols_to_use], on='TEAM_ID') 
                print "merged into %s" % k
            else:
                print "no merge match: %s, %s" % (k,v.columns) 
    return data

def crawlNBAGameData(args):
    print "crawling %s" % (args,)
    game_id, date, season = args

    url_template_summary = "http://stats.nba.com/stats/boxscoresummaryv2?GameID=%s"
    url_template_traditional = ("http://stats.nba.com/stats/boxscoretraditionalv2?"
                                "EndPeriod=10&EndRange=28800&GameID=%s&RangeType=0&"
                                "Season=%s&StartPeriod=1&StartRange=0")
    url_template_advanced = ("http://stats.nba.com/stats/boxscoreadvancedv2?EndPeriod=10&"
                             "EndRange=28800&GameID=%s&RangeType=0&Season=%s&"
                             "StartPeriod=1&StartRange=0")
    url_template_misc = ("http://stats.nba.com/stats/boxscoremiscv2?EndPeriod=10&"
                         "EndRange=28800&GameID=%s&RangeType=0&Season=%s"
                         "&StartPeriod=1&StartRange=0")
    url_template_scoring = ("http://stats.nba.com/stats/boxscorescoringv2?EndPeriod=10&"
                            "EndRange=28800&GameID=%s&RangeType=0&Season=%s&"
                            "StartPeriod=1&StartRange=0")
    url_template_usage = ("http://stats.nba.com/stats/boxscoreusagev2?EndPeriod=10&"
                          "EndRange=28800&GameID=%s&RangeType=0&Season=%s&"
                          "StartPeriod=1&StartRange=0") 
    url_template_fourfactors = ("http://stats.nba.com/stats/boxscorefourfactorsv2?"
                                "EndPeriod=10&EndRange=28800&GameID=%s&RangeType=0&"
                                "Season=%s&StartPeriod=1&StartRange=0") 
    url_template_tracking = ("http://stats.nba.com/stats/boxscoreplayertrackv2?"
                             "EndPeriod=10&EndRange=55800&GameID=%s&RangeType=2&"
                             "Season=%s&StartPeriod=1&StartRange=0")
    all_results = {}
    # get summary stats
    url = url_template_summary % game_id
    all_results = updateGameData(url, all_results)
    # get traditional stats
    url = url_template_traditional % (game_id, season)
    all_results = updateGameData(url, all_results)
    # get advanced stats
    url = url_template_advanced % (game_id, season)
    all_results = updateGameData(url, all_results) 
    # get misc stats
    url = url_template_misc % (game_id, season)
    all_results = updateGameData(url, all_results)
    # get usage stats
    url = url_template_usage % (game_id, season)
    all_results = updateGameData(url, all_results)
    # get fourfactors stats
    url = url_template_fourfactors % (game_id, season)
    all_results = updateGameData(url, all_results)
    # get tracking stats
    url = url_template_tracking % (game_id, season)
    all_results = updateGameData(url, all_results)
    # get scoring stats
    url = url_template_scoring % (game_id, season)
    all_results = updateGameData(url, all_results)

    # validate stats (if invalid, warn and continue)
    player_stat_shape = all_results["PlayerStats"]
    team_stat_shape = all_results["TeamStats"]

    if len(all_results['PlayerStats'].index) == 0:
        print "NO PLAYER STAT ROWS: %s" % (args,)
        return False
    if len(all_results['TeamStats'].index) != 2:
        print "TEAM STAT ROWS NOT EQUAL 2: %s" % (args,)
        return False

    # make sure players are in database
    players = all_results["PlayerStats"]['PLAYER_ID']
    for pid in players:
        player_data = findNBAPlayer(pid, crawl=True)

    # make sure teams are in db
    year = int(season.split('-')[0])
    teams = all_results['TeamStats']['TEAM_ID']
    for tid in teams:
        try:
            team_data = findNBATeam(tid, year=year, crawl=True)
        except Exception as e:
            # OK if we fail (there are some foreign games, e.g. Moscow CSKA games)
            print e

    # save it
    all_results = {k: v.to_json() for k,v in all_results.iteritems()}
    all_results['_id'] = game_id
    all_results['season'] = season
    all_results['teams'] = list(teams.values)
    all_results['players'] = list(players)
    all_results['date'] = date
    try:
        nba_conn.updateDocument(nba_games_collection, game_id, all_results, upsert=True)
    except DuplicateKeyError as e:
        pass
    return game_id

def findNBAPlayer(pid, crawl=True):
    query = {"_id": pid}
    result = nba_conn.findOne(nba_players_collection, query=query)
    if bool(result):
        return result
    elif crawl:
        crawled = crawlNBAPlayer(pid)
        if crawled:
            result = nba_conn.findOne(nba_players_collection, query=query)
            if bool(result):
                return result
    raise Exception("Could not find player %s" % pid)

def findNBATeam(tid, year, crawl=True):
    query = {"team_id": tid, "season": int(year)}
    result = nba_conn.findOne(nba_teams_collection, query=query)
    if bool(result):
        return result
    elif crawl:
        crawled = crawlNBATeam(tid, years=[year])
        if crawled:
            result = nba_conn.findOne(nba_teams_collection, query=query)
            if bool(result):
                return result
    raise Exception("Could not find team %s" % (query,))


def crawlNBATeam(tid, years=None):
    '''
    "http://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=2015-16&TeamID=1610612749"
    '''
    url_template = "http://stats.nba.com/stats/teaminfocommon?LeagueID=00&SeasonType=Regular+Season&TeamID=%s&season=%s"
    if not years:
        years = years_back
    for yr in years:
        season = createSeasonKey(yr)
        url = url_template % (tid, season)
        try: 
            resultsets = turnJSONtoPD(url)
        except Exception as e:
            raise Exception("Crawling team %s for year %s failed: %s, %s" % (tid, season, url, e))
        try:
            infoset = resultsets["TeamInfoCommon"]
            team_info = dict(infoset.iloc[0])
        except Exception as e:
            raise Exception("Crawling team %s for year %s parsing failed: %s, %s" % (tid, season, url, e))
        team_info['team_id'] = team_info.pop("TEAM_ID")
        team_info['season'] = yr
        try:
            nba_conn.saveDocument(nba_teams_collection, team_info)
        except DuplicateKeyError as e:
            pass
    return True


def updateRosters(update_players=True):
    now = datetime.now()
    start_year = now.year-1 if (now < datetime(day=1, month=10, year=now.year)) else now.year
    season = createSeasonKey(start_year)
    url_template = "http://stats.nba.com/stats/commonteamroster?LeagueID=00&Season=%s&TeamID=%s"
    teams = nba_teams_collection.find()
    for t in teams:
        tid = t['_id']
        url = url_template % (season, tid)
        try: 
            resultsets = turnJSONtoPD(url)
        except Exception as e:
            print "Crawling ROSTER for team %s failed: %s" % (tid, e)
            continue
        rosterset = resultsets['CommonTeamRoster']
        players = [_['PLAYER_ID'] for i, _ in rosterset.iterrows()]
        if update_players:
            for pid in players:
                crawlNBAPlayer(pid)
        nba_conn.updateDocument(nba_teams_collection, tid, {"roster": players}, upsert=False)

def crawlNBAPlayer(pid):
    url = "http://stats.nba.com/stats/commonplayerinfo?PlayerID=%s" % (pid,)

    try: 
        resultsets = turnJSONtoPD(url)
    except:
        print "Crawling player failed: %s" % pid
        return False
    info = resultsets.get("CommonPlayerInfo")
    if info is None:
        print "No player crawled: %s" % pid
        return False
    try:
        player_data = dict(info.iloc[0])
        pid = int(player_data.pop("PERSON_ID"))
        player_data['BR_POSITION'] = []

        # crawl rotowires
        '''
        wires = crawlPlayerRotowire(pid)
        player_data['wires'] = wires
        '''

        # crawl BR position
        player_name = player_data["DISPLAY_FIRST_LAST"].strip()
        birthday = player_data["BIRTHDATE"].strip()
        if birthday != "":
            birthday = dateutil.parser.parse(birthday)
            results = crawlBRPlayerPosition(pid, player_name)
            matched = False

            # if birthday is 1/1/1900, clearly an error, and take first result if there is only one result
            if birthday == datetime(year=1900,month=1,day=1) and len(results) == 1:
                print "BAD BIRTHDAY FOR %s: %s, setting to only result" % (player_name, birthday)
                bd, pos = results.pop(0)
                player_data['BR_POSITION'] = pos

            # try to match birthdays
            for bd, pos in results:
                possible_error = bd
                possible_error_2 = datetime(year=birthday.year, month=bd.month, day=bd.day)
                try:
                    possible_error = datetime(year=bd.year, month=bd.day, day=bd.month) # sometimes month and day flipped
                except Exception as e:
                    pass
                if abs(bd - birthday).days <= 120 or birthday == possible_error or birthday == possible_error_2:
                    print "Found BR Position for %s: %s" % (player_name, pos)
                    player_data['BR_POSITION'] = pos
                    matched = True
                    break

            if len(results) > 0 and not matched:
                print "Found BR results, but birthday doesn't match for %s, %s: %s vs %s" % (pid, player_name, birthday, results)
        else:
            print "No BD for %s, can't crawl BR" % (pid)

        nba_conn.updateDocument(nba_players_collection, pid, player_data, upsert=True)
        return True
    except Exception as e:
        print "Could not save nba player %s: %s" % (pid, e)
        print traceback.print_exc()
    return False

def createSeasonKey(yr):
    start_year = int(yr)
    end_year = start_year+1
    season_start = datetime(year=start_year, month=1, day=1)
    season_end = datetime(year=end_year, month=1, day=1)
    Season = "%s-%s" % (season_start.strftime("%Y"), season_end.strftime("%y"))
    return Season

def crawlNBAPlayerData():
    pool = mp.Pool(processes=8)
    def args():
        players = nba_players_collection.find({})
        for p in players:
            player_id = p['_id']
            fromyr = p['FROM_YEAR']
            toyr = p['TO_YEAR']
            if fromyr is None or toyr is None:
                print "Can't determine player year range"
                continue
                #valid_years = active_years
            fromyr  = int(fromyr)
            toyr = int(toyr)
            active_years = range(fromyr, toyr+1)
            valid_years = sorted(list(set(active_years) & set(years_back)))
            for yr in valid_years:
                yield (player_id, yr)
    for i, _ in enumerate(pool.imap_unordered(runPlayerCrawlFunctions, args()), 1): 
        pass
    pool.close()
    pool.join()


def runPlayerCrawlFunctions(args):
    player_id, yr = args
    print "Crawling %s for year %s" % (player_id, yr)
    # crawl player info and player shot chart data
    crawlNBAShotChart(player_id, yr)
    crawlNBAShot(player_id, yr)
    crawlNBADefense(player_id, yr)
    crawlNBARebound(player_id, yr)
    crawlNBAPass(player_id, yr)


def crawlNBAPlayerInfo(PID):
    url_template = "http://stats.nba.com/stats/commonplayerinfo?LeagueID=00&PlayerID=%s&SeasonType=Regular+Season"
    info_url = url_template % PID
    response = requests.get(info_url)
    headers = response.json()['resultSets'][0]['headers']
    info = response.json()['resultSets'][0]['rowSet']
    info_df = pd.DataFrame(info, columns=headers)
    return info_df


def crawlNBAShotChart(PID, year):
    url_template = ("http://stats.nba.com/stats/shotchartdetail?CFID=33&CFPARAMS=%s&"
                    "ContextFilter=&ContextMeasure=FGA&DateFrom=&DateTo=&GameID=&GameSegment=&LastNGames=0&"
                    "LeagueID=00&Location=&MeasureType=Base&Month=0&OpponentTeamID=0&Outcome=&PaceAdjust=N&"
                    "PerMode=PerGame&Period=0&PlayerID=%s&PlusMinus=N&Position=&Rank=N&RookieYear=&"
                    "Season=%s&SeasonSegment=&SeasonType=Regular+Season&TeamID=0&VsConference=&"
                    "VsDivision=&mode=Advanced&showDetails=0&showShots=1&showZones=0")
    Season = createSeasonKey(year)
    shot_chart_url = url_template % (Season, PID, Season)
    print shot_chart_url
    resultsets = turnJSONtoPD(shot_chart_url)
    shot_df = resultsets.get("Shot_Chart_Detail")
    if shot_df is None or shot_df.size == 0:
        print "No shot chart data found"
        return
    for idx, row in shot_df.iterrows():
        period = int(row['PERIOD'])
        min_rem = int(row['MINUTES_REMAINING'])
        sec_rem = int(row['SECONDS_REMAINING'])
        ts = "%s-%s:%s" % (period, min_rem, sec_rem)
        data = dict(row)
        data['time'] = ts
        data['player_id'] = data.pop('PLAYER_ID')
        data['game_id'] = data.pop('GAME_ID')
        try:
            nba_conn.saveDocument(shot_chart_collection, data)
        except DuplicateKeyError as e:
            continue
    return shot_df

def crawlNBAShot(PID, year):
    url_template = ("http://stats.nba.com/stats/playerdashptshotlog?DateFrom=&DateTo=&"
                    "GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&OpponentTeamID=0&"
                    "Outcome=&Period=0&PlayerID=%s&Season=%s&SeasonSegment=&"
                    "SeasonType=Regular+Season&TeamID=0&VsConference=&VsDivision=")
    Season = createSeasonKey(year)
    shot_chart_url = url_template % (PID, Season)
    print shot_chart_url
    resultsets = turnJSONtoPD(shot_chart_url)
    shot_df = resultsets.get("PtShotLog")
    if shot_df is None or shot_df.size == 0:
        print "No shot data found"
        return
    # get br player id, br game id, and parse gametime, then save
    for idx, row in shot_df.iterrows():
        period = int(row['PERIOD'])
        game_clock = row['GAME_CLOCK'].split(':')
        min_rem = int(game_clock[0])
        sec_rem = int(game_clock[1])
        ts = "%s-%s:%s" % (period, min_rem, sec_rem)
        data = dict(row)
        data['time'] = ts
        data['player_id'] = int(PID)
        data['game_id'] = data.pop('GAME_ID')
        # convert closest defender id
        def_id = row['CLOSEST_DEFENDER_PLAYER_ID']
        defender_data = findNBAPlayer(def_id, crawl=True)
        try:
            nba_conn.saveDocument(shot_collection, data)
        except DuplicateKeyError as e:
            continue
    return shot_df

def crawlNBADefense(PID, year):
    url_template = ("http://stats.nba.com/stats/playerdashptshotdefend?"
                    "DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&"
                    "Location=&Month=0&OpponentTeamID=0&Outcome=&PerMode=PerGame&Period=0&"
                    "PlayerID=%s&Season=%s&SeasonSegment=&SeasonType=Regular+Season&"
                    "TeamID=0&VsConference=&VsDivision=")
    Season = createSeasonKey(year)
    shot_chart_url = url_template % (PID, Season)
    print shot_chart_url
    resultsets = turnJSONtoPD(shot_chart_url)
    data = {k:v.to_json() for k,v in resultsets.iteritems() if v.size > 0}
    if len(data) == 0:
        print "No defense data found"
        return

    data['player_id'] = int(PID)
    data['year'] = year
    try:
        nba_conn.saveDocument(defense_collection, data)
    except DuplicateKeyError as e:
        pass
    return data

def crawlNBARebound(PID, year):
    url_template = ("http://stats.nba.com/stats/playerdashptreb?"
                    "DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&"
                    "Month=0&OpponentTeamID=0&Outcome=&PerMode=PerGame&Period=0&"
                    "PlayerID=%s&Season=%s&SeasonSegment=&SeasonType=Regular+Season&"
                    "TeamID=0&VsConference=&VsDivision=")
    Season = createSeasonKey(year)
    shot_chart_url = url_template % (PID, Season)
    print shot_chart_url
    resultsets = turnJSONtoPD(shot_chart_url)
    
    data = {k:v.to_json() for k,v in resultsets.iteritems() if v.size > 0}
    if len(data) == 0:
        print "No rebound data found"
        return

    data['player_id'] = int(PID)
    data['year'] = year
    try:
        nba_conn.saveDocument(rebound_collection, data)
    except DuplicateKeyError as e:
        pass
    return data

def crawlNBAPass(PID, year):
    url_template = ("http://stats.nba.com/stats/playerdashptpass?"
                    "DateFrom=&DateTo=&GameSegment=&LastNGames=0&LeagueID=00&Location=&Month=0&"
                    "OpponentTeamID=0&Outcome=&PerMode=PerGame&Period=0&PlayerID=%s&Season=%s&"
                    "SeasonSegment=&SeasonType=Regular+Season&TeamID=0&VsConference=&VsDivision=")
    Season = createSeasonKey(year)
    shot_chart_url = url_template % (PID, Season)
    print shot_chart_url
    resultsets = turnJSONtoPD(shot_chart_url)

    data = {k:v.to_json() for k,v in resultsets.iteritems() if v.size > 0}
    if len(data) == 0:
        print "No rebound data found"
        return

    data['player_id'] = int(PID)
    data['year'] = year
    try:
        nba_conn.saveDocument(pass_collection, data)
    except DuplicateKeyError as e:
        pass
    return data

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

'''
def translatePlayerNames(player_list, player_birthdays=None):
    logname = 'player_translation_problems.txt'

    if player_birthdays is not None and len(player_birthdays) != len(player_list):
        raise Exception("Player birthdays must be same length and order as player names")
    player_dicts = defaultdict(list)
    unmatched = []
    matched = []
    problem = False
    for i, d in enumerate(player_list):
        query = createPlayerRegex(d)
        full_name_query = {'full_name' : {'$regex' : query}}
        nickname_query = {'nickname' : {'$regex' : query}}
        if player_birthdays is not None:
            bd = player_birthdays[i]
            if bd:
                full_name_query['born'] = bd
                nickname_query['born'] = bd
        else:
            bd = None
        full_matched = player_collection.find(full_name_query)
        nick_matched = player_collection.find(nickname_query)
        for match_obj in nick_matched: 
            player_dicts[d].append(match_obj)
        if len(player_dicts[d]) == 0:
            for match_obj in full_matched:
                player_dicts[d].append(match_obj)
        if len(player_dicts[d]) == 0:
            print "NO MATCH %s" % d
            problem = 'No Match'
        elif len(player_dicts[d]) > 1:
            print "AMBIGUITY %s" % d
            problem = 'Ambiguous'
        if problem:
            logdata = {'Name': d, 
                       'BD': bd,
                       'Error': problem}
            logItem(logname, "%s: %s" % (datetime.now(), logdata))
    return dict(player_dicts)

def createPlayerRegex(raw_string):
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
'''

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
    # explicitly crawl players
    pool = mp.Pool(processes=4)
    def streamPID():
        for p in nba_players_collection.find({"POSITION": ""}):
            yield p['_id']
    for i, _ in enumerate(pool.imap_unordered(crawlNBAPlayer, streamPID(), 1)):
        pass
    '''

    # crawl upcoming games
    '''
    today = datetime.now() + timedelta(int(1))
    gl_crawl = upcomingGameCrawler(date=[today])
    gl_crawl.crawlPage()
    crawlNBATrackingStats()
    '''

    '''
    # get depth chart
    saveNBADepthChart()
    '''

    '''
    args = (u'0021400585', datetime(2015, 1, 15, 0, 0), u'2014')
    crawlNBAGameData(args)
    sys.exit(1)
    '''

    
    # get games (+ players + teams)
    crawlNBAGames('10/01/2007','10/01/2015')
    

    '''
    # get player data
    crawlNBAPlayerData()
    '''

    '''
    # get rosters
    updateRosters()
    '''