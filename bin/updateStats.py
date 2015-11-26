'''
Calculate stat vectors for all games in database
'''
import multiprocessing as mp
from datetime import datetime, timedelta

from statsETL.bball.statsExtraction import getPlayerVector, getTeamVector
from statsETL.db.mongolib import *
from statsETL.bball.apiExtraction import colorExtractor, TeamAPIExtractor, PlayerAPIExtractor, PLAYER_TYPES_ALLOWED, TEAM_TYPES_ALLOWED

RECALCULATE=False
CACHE=True
WINDOW=10 # DON'T CHANGE!!!
DAYS_BEHIND=3
DAYS_AHEAD=3

def yieldTeamGames(pivotdate, ahead=False):
    if not ahead:
        start_date = pivotdate - timedelta(days=DAYS_BEHIND)
        end_date = pivotdate
    else:
        start_date = pivotdate
        end_date = pivotdate + timedelta(days=DAYS_AHEAD)
    query = {'date' : {"$gte": start_date, "$lt": end_date}}
    games = nba_games_collection.find(query, no_cursor_timeout=True)
    for g in games:
        gid = g['_id']
        teams = g['teams']
        # try to find team stats
        for tid in teams:
            args = (tid, gid, RECALCULATE, WINDOW, CACHE)
            yield args

def yieldPlayerGames(pivotdate, ahead=False):
    if not ahead:
        start_date = pivotdate - timedelta(days=DAYS_BEHIND)
        end_date = pivotdate
    else:
        start_date = pivotdate
        end_date = pivotdate + timedelta(days=DAYS_AHEAD)
    query = {'date' : {"$gte": start_date, "$lt": end_date}}
    games = nba_games_collection.find(query, no_cursor_timeout=True)
    for g in games:
        gid = g['_id']
        player_teams = g['player_teams']
        date = g['date']
        for pid, tid in player_teams.iteritems():
            args = (int(pid), tid, gid, RECALCULATE, WINDOW, CACHE) 
            yield args

def getPlayerVector_worker(args):
    pid, tid, gid, recalculate, window, cache = args
    try:
        v = getPlayerVector(pid, tid, gid, recalculate=recalculate, window=window)
        header, row = PlayerAPIExtractor().extractVectors(v, PLAYER_TYPES_ALLOWED, cache=cache)
        return row
    except Exception as e:
        print "PLAYER VECTOR ERROR (%s, %s, %s): %s" % (pid, tid, gid, e)

def getTeamVector_worker(args):
    tid, gid, recalculate, window, cache = args
    try:
        v = getTeamVector(tid, gid, recalculate=recalculate, window=window)
        header, row = TeamAPIExtractor().extractVectors(v, TEAM_TYPES_ALLOWED, cache=cache)
        return row
    except Exception as e:
        print "TEAM VECTOR ERROR (%s, %s): %s" % (tid, gid, e)

def updateTeamStats(ahead=False):
    color = colorExtractor('team')
    now = datetime.now()
    pool = mp.Pool(processes=6)
    for i, response in enumerate(pool.imap_unordered(getTeamVector_worker, yieldTeamGames(now, ahead)), 1): 
        if response is None:
            continue
        color.updateColorRange(response)
    pool.close()
    pool.join()
    if not ahead:
        color.saveColorRange(now)

def updatePlayerStats(ahead=False):
    color = colorExtractor('player')
    now = datetime.now()
    pool = mp.Pool(processes=6)
    for i, response in enumerate(pool.imap_unordered(getPlayerVector_worker, yieldPlayerGames(now, ahead)), 1): 
        if response is None:
            continue
        color.updateColorRange(response)
    pool.close()
    pool.join()
    # save color range if we weren't looking ahead
    if not ahead:
        color.saveColorRange(now)

if __name__ == "__main__":
    # args = (201601, 1610612746, '0021500052', RECALCULATE, WINDOW, CACHE) 
    # getPlayerVector_worker(args)
    # sys.exit(1)

    updateTeamStats(ahead=False)
    #updateTeamStats(ahead=True)
    updatePlayerStats(ahead=False)
    #updatePlayerStats(ahead=True)
