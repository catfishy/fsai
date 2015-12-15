'''
Calculate stat vectors for all games in database
'''
import multiprocessing as mp
from datetime import datetime, timedelta
import random
from collections import defaultdict
from itertools import cycle, islice

from statsETL.bball.statsExtraction import getPlayerVector, getTeamVector
from statsETL.db.mongolib import *
from statsETL.bball.apiExtraction import colorExtractor, TeamAPIExtractor, PlayerAPIExtractor, PLAYER_TYPES_ALLOWED, TEAM_TYPES_ALLOWED

RECALCULATE=True
CACHE=True
WINDOW=10 # DON'T CHANGE!!!
DAYS_BEHIND=3
DAYS_AHEAD=3

def roundrobin(*iterables):
    '''
    roundrobin('ABC', 'D', 'EF') --> A D E B F C
    '''
    # Recipe credited to George Sakkis
    pending = len(iterables)
    nexts = cycle(iter(it).next for it in iterables)
    while pending:
        try:
            for next in nexts:
                yield next()
        except StopIteration:
            pending -= 1
            nexts = cycle(islice(nexts, pending))

def yieldTeamGames(pivotdate, ahead=False):
    if not ahead:
        start_date = pivotdate - timedelta(days=DAYS_BEHIND)
        end_date = pivotdate
    else:
        start_date = pivotdate
        end_date = pivotdate + timedelta(days=DAYS_AHEAD)
    query = {'date' : {"$gte": start_date, "$lt": end_date}}
    games = nba_games_collection.find(query, no_cursor_timeout=True)
    all_args = defaultdict(list)
    counter = 0
    for g in games:
        gid = g['_id']
        teams = g['teams']
        date = g['date']
        # try to find team stats
        for tid in teams:
            args = (tid, gid, RECALCULATE, WINDOW, CACHE)
            all_args[date].append(args)
            counter += 1
    print 'TEAM GAMES TO CALCULATE: %s' % counter
    sorted_dates = sorted(all_args.keys())
    for date in sorted_dates:
        for arg in all_args[date]:
            yield arg

def yieldPlayerGames(pivotdate, ahead=False):
    if not ahead:
        start_date = pivotdate - timedelta(days=DAYS_BEHIND)
        end_date = pivotdate
    else:
        start_date = pivotdate
        end_date = pivotdate + timedelta(days=DAYS_AHEAD)
    query = {'date' : {"$gte": start_date, "$lt": end_date}}
    games = nba_games_collection.find(query, no_cursor_timeout=True)
    # gather args
    all_args = defaultdict(dict)
    counter = 0
    for g in games:
        gid = g['_id']
        player_teams = g['player_teams']
        date = g['date']
        for pid, tid in player_teams.iteritems():
            args = (int(pid), tid, gid, RECALCULATE, WINDOW, CACHE) 
            if tid in all_args[date]:
                all_args[date][tid].append(args)
            else:
                all_args[date][tid] = [args]
            counter += 1
    print 'PLAYER GAMES TO CALCULATE: %s' % counter
    # within date, interweave team players
    sorted_dates = sorted(all_args.keys())
    for date in sorted_dates:
        by_team_args = all_args[date].values()
        for arg in roundrobin(*by_team_args):
            yield arg


def getPlayerVector_worker(args):
    pid, tid, gid, recalculate, window, cache = args
    try:
        v = getPlayerVector(pid, tid, gid, recalculate=recalculate, window=window)
        row = PlayerAPIExtractor().extractVectors(v, PLAYER_TYPES_ALLOWED, cache=cache)
        return row
    except Exception as e:
        print "PLAYER VECTOR ERROR (%s, %s, %s): %s" % (pid, tid, gid, e)

def getTeamVector_worker(args):
    tid, gid, recalculate, window, cache = args
    try:
        v = getTeamVector(tid, gid, recalculate=recalculate, window=window)
        row = TeamAPIExtractor().extractVectors(v, TEAM_TYPES_ALLOWED, cache=cache)
        return row
    except Exception as e:
        print "TEAM VECTOR ERROR (%s, %s): %s" % (tid, gid, e)

def updateTeamStats(ahead=False):
    color = colorExtractor('team')
    now = datetime.now()
    pool = mp.Pool(processes=8)
    for i, response in enumerate(pool.imap(getTeamVector_worker, yieldTeamGames(now, ahead)), 1): 
        print "TEAM STATS COUNT: %s" % (i+1)
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
    pool = mp.Pool(processes=8)
    for i, response in enumerate(pool.imap(getPlayerVector_worker, yieldPlayerGames(now, ahead)), 1): 
        print "PLAYER STATS COUNT: %s" % (i+1)
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
    updateTeamStats(ahead=True)
    updatePlayerStats(ahead=False)
    updatePlayerStats(ahead=True)
