'''
Functions necessary to power API endpoints,
as well as feature vector objects for player-in-game
'''

from datetime import datetime, timedelta
from collections import defaultdict
import copy
import itertools
import traceback
import csv
import os
import multiprocessing as mp
import re

import numpy as np
from scipy.spatial import distance
import matplotlib.pyplot as plt

from statsETL.db.mongolib import *

def playerOffenseScatter(player_vectors):
    pass

def playerDefenseScatter(player_vectors):
    pass

def playerShotChartScatter(player_vectors):
    pass

def teamOffenseScatter(team_vectors):
    pass

def teamDefenseScatter(team_vectors):
    pass

def teamShotChartScatter(team_vectors):
    pass

class teamFeatureVector(object):

    '''
    team performance up until a certain day, given a stats window
    subsumed by playerFeatureVector
    '''

    def __init__(self, tid, end_date, window=5):
        self.tid = tid
        self.gid = gid

    def loadStats():
        pass


class playerFeatureVector(object):

    def __init__(self, pid, gid, team_vectors=None):
        self.pid = pid
        self.gid = gid

        # option to load in team vectors instead of calculating them
        if isinstance(team_vectors, dict):
            self.team_vectors = team_vectors
        else:
            self.team_vectors = None

        self.loadGame()
        self.loadStats()

    def loadGame():
        '''
        Load teams, players
        '''
        pass

    def loadStats():
        '''
        Load team stats if necessary (own, ownopp, opp, oppopp)
        Load individual stats
        Load additional misc stats
        '''
        pass

    def statsForPlayerInGame(pid, gid):
        pass

    def parsePlayerShotChart(pid, gid):
        pass

    def statsForTeamInGame(tid, gid):
        pass

    def parseTeamShotChart(tid, gid):
        pass