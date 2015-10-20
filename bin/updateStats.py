'''
Calculate stat vectors for all games in database
'''
import multiprocessing as mp

from statsETL.bball.statsExtraction import getPlayerVector, getTeamVector
from statsETL.db.mongolib import *

RECALCULATE=False
WINDOW=10

def yieldTeamGames():
    games = nba_games_collection.find(no_cursor_timeout=True)
    for g in games:
        gid = g['_id']
        teams = g['teams']
        # try to find team stats
        for tid in teams:
            args = (tid, gid, RECALCULATE, WINDOW)
            print args
            yield args


def yieldPlayerGames():
    games = nba_games_collection.find(no_cursor_timeout=True)
    for g in games:
        gid = g['_id']
        player_teams = g['player_teams']
        for pid, tid in player_teams.iteritems():
            pid = int(pid)
            args = (pid, tid, gid, RECALCULATE, WINDOW) 
            print args
            yield args


def getPlayerVector_worker(args):
    pid, tid, gid, recalculate, window = args
    try:
        getPlayerVector(pid, tid, gid, recalculate=recalculate, window=window)
    except Exception as e:
        print "PLAYER VECTOR ERROR (%s, %s, %s): %s" % (pid, tid, gid, e)

def getTeamVector_worker(args):
    tid, gid, recalculate, window = args
    try:
        getTeamVector(tid, gid, recalculate=recalculate, window=window)
    except Exception as e:
        print "TEAM VECTOR ERROR (%s, %s): %s" % (tid, gid, e)

def updateTeamStats():
    pool = mp.Pool(processes=8)
    for i, _ in enumerate(pool.imap_unordered(getTeamVector_worker, yieldTeamGames()), 1): 
        pass
    pool.close()
    pool.join()

def updatePlayerStats():
    pool = mp.Pool(processes=8)
    for i, _ in enumerate(pool.imap_unordered(getPlayerVector_worker, yieldPlayerGames()), 1): 
        pass
    pool.close()
    pool.join()

if __name__ == "__main__":
    updateTeamStats()
