"""
Version 1:

- Find out what the upcoming games are
- Find out which players are likely to play which positions by team roster, player positions, and previous minutes played
- For the stats that matter (get points), calculate trajectory for each player in last 15 games
- ? do projections somehow (incorporating matchup context)

"""
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np

from statsETL.db.mongolib import *


def findAllTrainingGames(pid, limit=40, time=None, min_count=None):
    '''
    TODO: FILTER games by player's current team????? (or just general teammate context)
    '''
    if time is None:
        time = datetime.now()
    # find all training games
    playergames = list(player_game_collection.find({"player_id": pid, "game_time" : {"$lt" : time}}, sort=[("game_time",-1)]))
    # drop games where mp == 0 and 
    valid_games = [g for g in playergames if g['MP'] != 0.0]
    # limit
    valid_games = valid_games[:limit]
    # check above min
    if min_count and len(valid_games) < min_count:
        raise Exception("Couldn't find more than %s games for %s" % (min_count, pid))
    return valid_games



class NBAFeatureExtractor(object):

    PLAYER_STATS = ["STL%", "FT", "3P", "TOV", "FG", "3PA", "DRB", "ORB%", 
                    "BLK%", "AST", "FT%", "3PAr", "PF", "PTS", "FGA", "FG%", 
                    "STL", "TRB", "TOV%", "AST%", "FTA", "eFG%", "BLK", "FTr", 
                    "+/-", "USG%", "DRB%", "TS%", "MP", "DRtg", "ORtg", "TRB%", 
                    "ORB", "3P%"]
    PER48_STATS = ["FT", "3P", "TOV", "FG", "3PA", "DRB", "AST", "PF", 
                   "PTS", "FGA", "STL", "TRB", "FTA", "BLK", "ORB"]
    TEAM_STATS = ["TOV%", "Pace", "ORtg", "FT/FGA", "ORB%", "eFG%",  "T"]
    ESPN_TEAM_STATS = ['ESPN_ORR', 'ESPN_AST', 'ESPN_PACE', 'ESPN_TS%', 
                       'ESPN_EFF FG%', 'ESPN_TO', 'ESPN_DRR', 'ESPN_REBR', 
                       'ESPN_DEF EFF', 'ESPN_OFF EFF']
    INTENSITY_STATS = ['own_win_rate','opp_win_rate','own_streak','opp_streak',
                       'own_aways_in_6','opp_aways_in_6','mp_in_6']
    CAT_ONE_HOT_ON = 1
    CAT_ONE_HOT_OFF= -1

    def __init__(self, team_id, target_game=None):
        if isinstance(target_game, str):
            self.target_game_id = target_game
            self.upcoming_game = None
        elif isinstance(target_game, dict):
            self.upcoming_game = target_game
            self.target_game_id = None
        else:
            raise Exception("target_game not a game id or a game dict")

        self.own_team_id = team_id
        self.opp_team_id = None
        self.season_start = None

        if self.target_game_id:
            self.target_game_info = game_collection.find_one({"_id":self.target_game_id})
            if not self.target_game_info:
                raise Exception("Could not find game %s" % self.target_game_id)

            self.ts = self.target_game_info['time']

            # check that team is in the game
            self.team_game_info = team_game_collection.find_one({"game_id" : self.target_game_id,
                                                                   "team_id" : self.own_team_id})
            if not self.team_game_info:
                raise Exception("Could not find team in game")
            self.opp_team_id = self.target_game_info['away_id'] if self.own_team_id == self.target_game_info['home_id'] else self.target_game_info['home_id']
            self.home_id = self.target_game_info['home_id']
            self.away_id = self.target_game_info['away_id']
        elif self.upcoming_game:
            self.ts = self.upcoming_game['time']
            self.target_game_info = None
            self.team_game_info = None
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            # try to look for game
            if self.ts < datetime.now():
                found = game_collection.find_one({"time": self.ts, "teams" : [team1_id, team2_id]})
                self.target_game_info = found if found else None
            self.opp_team_id = team2_id if (self.own_team_id == team1_id) else team1_id
            self.home_id = team1_id
            self.away_id = team2_id
        # grab team rows
        self.own_team = team_collection.find_one({"_id": self.own_team_id})
        self.opp_team = team_collection.find_one({"_id": self.opp_team_id})
        self.home_team = self.own_team if (self.own_team_id == self.home_id) else self.opp_team


        # get start of season based on game timestamp (use most recent 8/1)
        self.season_start = datetime(year=self.ts.year, month=8, day=1)
        if self.season_start > self.ts:
            self.season_stat = datetime(year=self.ts.year-1, month=8, day=1)

        # get all teams and locations
        results = team_collection.find({})
        self.all_teams = set()
        self.all_locations = set()
        for r in results:
            self.all_teams.add(r['_id'])
            self.all_locations.add(r['location'])
        self.all_teams = list(self.all_teams)
        self.all_locations = list(self.all_locations)

    def timestamp(self):
        return self.ts

    def convertPlayerGameStatsToPer48(self, stats):
        new_stats = []
        for s in stats:
            mp = float(s['MP'])
            if mp == 0.0:
                new_stats.append(s)
            else:
                ratio = 48.0 / mp
                for k in self.PER48_STATS:
                    s[k] = s[k] * ratio
                new_stats.append(s)
        return new_stats

    def averageStats(self, stats, allowed_keys):
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            values = [x for x in v if x is not None]
            if len(values) > 0:
                trajectories[k] = np.mean(values)
            else:
                trajectories[k] = np.nan
        return trajectories

    def serializeFeatures(self, cont_labels, cat_labels, data):
        cont_features = [data.get(k, np.nan) for k in cont_labels]
        cont_features = [v if v is not None else np.nan for v in cont_features]
        cat_features = [data.get(k, self.CAT_ONE_HOT_OFF) for k in cat_labels]
        cat_features = [v if v is not None else self.CAT_ONE_HOT_OFF for v in cat_features]
        return cont_features, cat_features

    def getPlayerStats(self):
        '''
        for each team, find the starting lineup and the backup lineup (10 players total, 10 players with most minutes)
        get player stats for each player, including:
            - espn tracking stats
            - running avg stats
            - season avg stats
            - height, weight, experience, handedness


        -> FIND STARTERS, AND 2ND STRING
        -> IF ANALYZING PREVIOUS GAME, EXCLUDE PLAYERS WHO DIDN'T PLAY
        -> ORGANIZE BY STARTING LINEUP + BENCH LINEUP

        '''
        # find lineups if possible
        if self.target_game_info:
            home_starters = self.target_game_info['home_starters']
            away_starters = self.target_game_info['away_starters']
            # determine positions
            home_starter_positions = defaultdict(list)
            away_starter_positions = defaultdict(list)
            for s in home_starters:
                row = player_collection.find_one({"_id": s})
                for p in row['position']:
                    home_starter_positions[p].append(s)
            for s in away_starters:
                row = player_collection.find_one({"_id": s})
                for p in row['position']:
                    away_starter_positions[p].append(s)
            # decide lineup
            home_starting_lineup = {}
            home_starter_positions = dict(home_starter_positions)
            while len(home_starting_lineup.keys()) < 5:
                for k,v in home_starter_positions.iteritems():
                    # remove used
                    v_filtered = [_ for _ in v if _ not in home_starting_lineup.values()]
                    if len(v_filtered) == 1:
                        home_starting_lineup[k] = v_filtered[0]

            away_starting_lineup = {}
            away_starter_positions = dict(away_starter_positions)
            while len(away_starting_lineup.keys()) < 5:
                for k,v in away_starter_positions.iteritems():
                    # remove used
                    v_filtered = [_ for _ in v if _ not in away_starting_lineup.values()]
                    if len(v_filtered) == 1:
                        away_starting_lineup[k] = v_filtered[0]
            # choose who's home and who's away
            own_starting_lineup = home_starting_lineup if self.player_team == self.home_id else away_starting_lineup
            opp_starting_lineup = home_starting_lineup if self.opp_team == self.home_id else away_starting_lineup
        else:
            # find from espn, find the one saved immediately before self.ts
            depths = depth_collection.find({"time" : {"$lt" : self.ts}}, sort=[('time', -1)])
            depth_used = depths[0]['stats']
            own_starting_lineup = depth_used[self.player_team]
            opp_starting_lineup = depth_used[self.opp_team]

        # for own team
        own_player_stats = defaultdict(dict)
        for pid in self.own_team['players']:
            season_stats = list(player_game_collection.find({"player_id": pid, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, sort=[("game_time",-1)]))
            season_stats = self.convertPlayerGameStatsToPer48(season_stats)
            own_player_stats[pid]['season_avg'] = self.averageStats(season_stats, self.PLAYER_STATS)
            own_player_stats[pid]['running_avg'] = self.averageStats(season_stats[:5], self.PLAYER_STATS)
            own_player_stats[pid]['MP'] = own_player_stats[pid]['running_avg']['MP']
            # find closest espn player tracking stat row
            tracking = list(espn_player_stat_collection.find({"player_id": pid}))
            sorted_tracking = sorted(tracking, key=lambda x: abs(self.ts-x['time']))
            own_player_stats[pid]['espn_tracking'] = sorted_tracking[0]
            own_player_stats[pid]['player_row'] = player_collection.find_one({"_id": pid})

        # for opp team
        opp_player_stats = defaultdict(dict)
        for pid in self.opp_team['players']:
            season_stats = list(player_game_collection.find({"player_id": pid, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, sort=[("game_time",-1)]))
            season_stats = self.convertPlayerGameStatsToPer48(season_stats)
            opp_player_stats[pid]['season_avg'] = self.averageStats(season_stats, self.PLAYER_STATS)
            opp_player_stats[pid]['running_avg'] = self.averageStats(season_stats[:5], self.PLAYER_STATS)
            opp_player_stats[pid]['MP'] = opp_player_stats[pid]['running_avg']['MP']
            # find closest espn player tracking stat row
            tracking = list(espn_player_stat_collection.find({"player_id": pid}))
            sorted_tracking = sorted(tracking, key=lambda x: abs(self.ts-x['time']))
            opp_player_stats[pid]['espn_tracking'] = sorted_tracking[0]
            opp_player_stats[pid]['player_row'] = player_collection.find_one({"_id": pid})

        top_ten_own = sorted(own_player_stats.keys(), key=lambda x: own_player_stats['MP'], reverse=True)[:10]
        top_ten_opp = sorted(opp_player_stats.keys(), key=lambda x: opp_player_stats['MP'], reverse=True)[:10]

        self.player_stats = {}

        for i, pid in enumerate(top_ten_own):
            self.player_stats['own_%s' % i] = own_player_stats[pid]
        for i, pid in enumerate(top_ten_opp):
            self.player_stats['opp_%s' % i] = opp_player_stats[pid]

    def getTeamStats(self):
        '''
        query for all the necessary team stats, which includes:
            - espn team stats (closest to game time)
            - running 5 game team stat averages
            - season averages
            - most recent game played
            - previous matchups in the season
        '''
        # find previous matchups between the two teams
        self.prev_matchup_games = list(game_collection.find({"teams": [self.player_team, self.opp_team], "time": {"$gt" : self.season_start}}))
        self.own_prev_matchup_stats = [team_game_stat_collection.find_one({"team_id": self.own_team_id, "game_id": g['_id']}) for g in self.prev_matchup_games]
        self.opp_prev_matchup_stats = [team_game_stat_collection.find_one({"team_id": self.opp_team_id, "game_id": g['_id']}) for g in self.prev_matchup_games]

        # find espn team stats
        espnstat = list(espn_stat_collection.find())
        sorted_espnstat = sorted(espnstat, key=lambda x: abs(self.ts-x['time']))
        recent_espnstat = sorted_espnstat[0]['stats']
        self.own_espn_team_stats = recent_espnstat[self.own_team_id]
        self.opp_espn_team_stats = recent_espnstat[self.opp_team_id]

        # own season/running avg team stats ( + reflection)
        own_results = list(team_game_collection.find({"team_id": self.own_team_id, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.own_team_id)
        own_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$ne": self.own_team_id}}) for g in own_results]
        
        self.own_most_recent_game = game_collection.find_one({"_id" : own_results[0]['game_id']})
        self.own_last_10 = copy.deepcopy(own_results[:10])
        self.own_gametimes = [o['game_time'] for o in own_results[:5]]
        self.own_team_season_avgs = self.averageStats(own_results, self.TEAM_STATS)
        self.own_reflection_season_avgs = self.averageStats(own_reflection_results, self.TEAM_STATS)
        self.own_team_running_avgs = self.averageStats(own_results[:5], self.TEAM_STATS)
        self.own_reflection_running_avgs = self.averageStats(own_reflection_resultsp[:5], self.TEAM_STATS)

        # opp season/running avg team stats ( + reflection)
        opp_results = list(team_game_collection.find({"team_id": self.opp_team_id, "game_time": {"$lt": self.ts, "$gt": self.season_start}}, 
                                            sort=[("game_time",-1)]))
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % self.opp_team_id)

        opp_reflection_results = [team_game_collection.find_one({"game_id": g['game_id'], "team_id": {"$ne": self.opp_team_id}}) for g in opp_results]
        
        self.opp_most_recent_game = game_collection.find_one({"_id" : opp_results[0]['game_id']})
        self.opp_last_10 = copy.deepcopy(opp_results[:10])
        self.opp_gametimes = [o['game_time'] for o in opp_results]
        self.opp_team_season_avgs = self.averageStats(opp_results, self.TEAM_STATS)
        self.opp_reflection_season_avgs = self.averageStats(opp_reflection_results, self.TEAM_STATS)
        self.opp_team_running_avgs = self.averageStats(opp_results[:5], self.TEAM_STATS)
        self.opp_reflection_running_avgs = self.averageStats(opp_reflection_results[:5], self.TEAM_STATS)

    def runEncoderFeatures(self):
        """

        TEAM FEATURES (OWN,OPP):
            - espn team features (shortest temporal distance)
            - running avg team features (avg over 5 games, include running avgs of opponent teams faced)
            - record features (win percentage, aways in last 5, wins in last 10)

        OWN LINEUP (5 positions, 2 depth):
            - FOR EACH pid:
                - running avg player features
                - height, weight, experience, handedness
                - espn tracking stats

        OPP LINEUP (5 positions, 2 depth):
            - FOR EACH pid:
                - running avg player features
                - height, weight, experience, handedness
                - espn non-offensive tracking stats

        LOCATION FEATURES:
            - location
            - own team and opp team
            - b2b for own team and opp team

        PREVIOUS MATCHUP FEATURES (OWN, OPP):
            - same as TEAM FEATURES (without espn team features) but only averaging over previous team matchups

        TREND FEATURES:
            - for each player in own/opp lineup stats:
                - (season - running avg) for player stats
                - (season - matchup avg) for player stats
            - for each team:
                - (season - running avg) for team stats
                - (season - matchup avg) for team stats

        """
        self.getTeamStats()
        self.getPlayerStats()

        fns = [self.teamFeatures, self.lineupFeatures, self.locationFeatures, self.recordFeatures, self.matchupFeatures, self.trendFeatures]
        cat_labels_all = []
        cat_features_all = []
        cont_labels_all = []
        cont_features_all = []
        cat_feat_splits_all = []
        for fn in fns:
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fn()
            cat_labels_all += cat_labels
            cat_features_all += cat_features
            cont_labels_all += cont_labels
            cont_features_all += cont_features
            cat_feat_splits_all += cat_feat_splits

        return cat_labels_all, cat_features_all, cont_labels_all, cont_features_all, cat_feat_splits_all

    def teamFeatures(self):
        '''
        for each team: espn stats, running avg, running opp average
        '''
        data = {'own_espn_team_stats': copy.deepcopy(self.own_espn_team_stats),
                'opp_espn_team_stats': copy.deepcopy(self.opp_espn_team_stats),
                'own_team_running_avgs': copy.deepcopy(self.own_team_running_avgs),
                'opp_team_running_avgs': copy.deepcopy(self.opp_team_running_avgs),
                'own_team_reflect_avgs': copy.deepcopy(self.own_reflection_running_avgs),
                'opp_team_reflect_avgs': copy.deepcopy(self.opp_reflection_running_avgs)}

        # flatten data
        top_keys = data.keys()
        for k in top_keys:
            for k_2,v in data[k]:
                data["%s_%s" % k,k_2] = v
            data.pop(k,None)

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)


    def lineupFeatures(self):
        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)


    def locationFeatures(self):
        game_location = self.home_team['location']
        teams_one_hot = {x: self.CAT_ONE_HOT_OFF for x in self.all_teams}
        location_one_hot = {x: self.CAT_ONE_HOT_OFF for x in self.all_locations}

        teams_one_hot[self.opp_team_id] = 1
        teams_one_hot[self.own_team_id] = 1
        location_one_hot[game_location] = 1

        data = {}
        for k,v in teams_one_hot.iteritems():
            data['teams_%s' % k] = v
        for k,v in location_one_hot.iteritems():
            data['locations_%s' % k] = v

        game_day = datetime(year=self.ts.year,month=self.ts.month,day=self.ts.day)
        own_last_game_days = [datetime(year=x.year,month=x.month,day=x.day) for x in self.own_gametimes]
        opp_last_game_days = [datetime(year=x.year,month=x.month,day=x.day) for x in self.opp_gametimes]
        own_game_deltas = [(game_day - x).days for x in own_last_game_days]
        opp_game_deltas = [(game_day - x).days for x in opp_last_game_days]
        own_game_index = [0] * 4
        opp_game_index = [0] * 4
        for d in own_game_deltas:
            if d < 5:
                own_game_index[d-1] = 1
        for d in opp_game_deltas:
            if d < 5:
                opp_game_index[d-1] = 1

        data['own_b2b'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:1]) >= 1) else self.CAT_ONE_HOT_OFF
        data['own_2of3'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:2]) >= 1) else self.CAT_ONE_HOT_OFF
        data['own_3of4'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:3]) >= 2) else self.CAT_ONE_HOT_OFF
        data['own_4of5'] = self.CAT_ONE_HOT_ON if (sum(own_game_index[:4]) >= 3) else self.CAT_ONE_HOT_OFF

        data['opp_b2b'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:1]) >= 1) else self.CAT_ONE_HOT_OFF
        data['opp_2of3'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:2]) >= 1) else self.CAT_ONE_HOT_OFF
        data['opp_3of4'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:3]) >= 2) else self.CAT_ONE_HOT_OFF
        data['opp_4of5'] = self.CAT_ONE_HOT_ON if (sum(opp_game_index[:4]) >= 3) else self.CAT_ONE_HOT_OFF

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = ['teams_%s' % k for k in self.all_teams] + ['locations_%s' % k for k in self.all_locations]
        cat_feature_splits = [len(self.all_teams), len(self.all_locations)]
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

    def recordFeatures(self):
        '''
        win percentage, aways in last 5, last 10 record, streaks
        away or home for the own team
        '''
        self.opp_most_recent_game = opp_results[0]
        self.opp_last_10 = copy.deepcopy(opp_results[:10])

        # calculate number of aways in last 5
        own_aways = len([_ for _ in self.own_last_10[:5] if _.get('location') == 'Away'])
        opp_aways = len([_ for _ in self.opp_last_10[:5] if _.get('location') == 'Away'])

        # calculate last 10 wins
        own_last_10_wins = 0
        for g in self.own_last_10:
            pts = g['T']
            other_team_game = team_game_collection.find_one({"game_id" : g['game_id'], "team_id" : {"$ne" : g['team_id']}})
            other_pts = other_team_game['T']
            if pts > other_pts:
                own_last_10_wins += 1
        opp_last_10_wins = 0
        for g in self.opp_last_10:
            pts = g['T']
            other_team_game = team_game_collection.find_one({"game_id" : g['game_id'], "team_id" : {"$ne" : g['team_id']}})
            other_pts = other_team_game['T']
            if pts > other_pts:
                opp_last_10_wins += 1

        # parse records and streaks
        own_recent_loc = 'home' if (self.own_team['_id'] == self.own_most_recent_game['home_id']) else 'away'
        own_record = self.own_most_recent['%s_record' % own_recent_loc]
        own_streak = self.own_most_recent['%s_streak' % own_recent_loc]

        opp_recent_loc = 'home' if (self.opp_team['_id'] == self.opp_most_recent_game['home_id']) else 'away'
        opp_record = opp_most_recent['%s_record' % opp_recent_loc]
        opp_streak = opp_most_recent['%s_streak' % opp_recent_loc]

        current_location = 1 if (self.home_team['_id'] == self.own_team['_id']) else -1

        if own_record is None:
            own_winning_percent = 1.0
            own_streak = 0.0
        else:
            own_record_parts = own_record.split('-')
            own_wins = int(own_record_parts[0].strip())
            own_losses = int(own_record_parts[1].strip())
            own_winning_percent = own_wins / float(own_losses + own_wins)
            if 'Lost' in own_streak:
                own_streak = -int(own_streak.replace('Lost ','').strip())
            else:
                own_streak = int(own_streak.replace('Won ','').strip())
        if opp_record is None:
            opp_winning_percent = 1.0
            opp_streak = 0.0
        else:
            opp_record_parts = opp_record.split('-')
            opp_wins = int(opp_record_parts[0].strip())
            opp_losses = int(opp_record_parts[1].strip())
            opp_winning_percent = opp_wins / float(opp_losses + opp_wins)
            if 'Lost' in opp_streak:
                opp_streak = -int(opp_streak.replace('Lost ','').strip())
            else:
                opp_streak = int(opp_streak.replace('Won ','').strip())

        # load feature dictionary
        data = {'own_win_rate': own_winning_percent,
                'opp_win_rate': opp_winning_percent,
                'own_streak': own_streak,
                'opp_streak': opp_streak,
                'own_aways': own_aways,
                'opp_aways': opp_aways,
                'own_last_10_wins': own_last_10_wins,
                'opp_last_10_wins': opp_last_10_wins,
                'own_home_away': current_location, # TEAM-SUBJECTIVE STAT
                }

        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)

    def trendFeatures(self):
        # serialize
        cont_labels = list(sorted(data.keys()))
        cat_labels = []
        cat_feature_splits = []
        cont_features, cat_features = self.serializeFeatures(cont_labels, cat_labels, data)

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feature_splits)




class featureExtractor(object):

    """
    Extracts features for a player in a game

        + recent basic/advanced allowed stats of opposing team (allowed and earned)
        + recent basic/advanced allowed stats of position player on opposition
        + recent basic/advanced stats for player
        + game location, opposing team
        + b2b, 2/3, 3/4
    """

    PLAYER_STATS = ["STL%", "FT", "3P", "TOV", "FG", "3PA", "DRB", "ORB%", 
                    "BLK%", "AST", "FT%", "3PAr", "PF", "PTS", "FGA", "FG%", 
                    "STL", "TRB", "TOV%", "AST%", "FTA", "eFG%", "BLK", "FTr", 
                    "+/-", "USG%", "DRB%", "TS%", "MP", "DRtg", "ORtg", "TRB%", 
                    "ORB", "3P%"]
    PER48_STATS = ["FT", "3P", "TOV", "FG", "3PA", "DRB", "AST", "PF", 
                   "PTS", "FGA", "STL", "TRB", "FTA", "BLK", "ORB"]
    TEAM_STATS = ["TOV%", "Pace", "ORtg", "FT/FGA", "ORB%", "eFG%",  "T"]
    ESPN_TEAM_STATS = ['ESPN_ORR', 'ESPN_AST', 'ESPN_PACE', 'ESPN_TS%', 
                       'ESPN_EFF FG%', 'ESPN_TO', 'ESPN_DRR', 'ESPN_REBR', 
                       'ESPN_DEF EFF', 'ESPN_OFF EFF']
    INTENSITY_STATS = ['own_win_rate','opp_win_rate','own_streak','opp_streak',
                       'win_rate_diff','own_aways_in_6','opp_aways_in_6','mp_in_6']


    def __init__(self, player_id, target_game=None):
        self.pid = str(player_id)
        if not self.pid:
            raise Exception("Must specify a player")
        self.player_info = player_collection.find_one({"_id": self.pid})
        if not self.player_info:
            raise Exception("Could not find player %s" % self.pid)
        if isinstance(target_game, str):
            self.target_game_id = target_game
            self.upcoming_game = None
        elif isinstance(target_game, dict):
            self.upcoming_game = target_game
            self.target_game_id = None
        else:
            raise Exception("target_game not a game id or a game dict")
        if self.target_game_id:
            # get the game
            self.target_game_info = game_collection.find_one({"_id":self.target_game_id})
            if not self.target_game_info:
                raise Exception("Could not find game %s" % self.target_game_id)
            self.ts = self.target_game_info['time']
            # check that player is in the game
            self.player_game_info = player_game_collection.find_one({"game_id" : self.target_game_id,
                                                                     "player_id" : self.pid})
            if not self.player_game_info:
                raise Exception("Could not find player in game")
            self.player_team = self.player_game_info["player_team"]
            # find team
            self.team = team_collection.find_one({"_id": self.player_team})
            # find team game info
            self.team_game_info = team_game_collection.find_one({"team_id" : self.player_team,
                                                                 "game_id": self.target_game_id})
            if self.player_team == self.target_game_info['home_id']:
                self.opp_team = self.target_game_info['away_id']
            else:
                self.opp_team = self.target_game_info['home_id']
        else:
            self.target_game_info = None
            self.player_game_info = None
            self.team_game_info = None
            # get player team
            self.team = list(team_collection.find({"players" : self.pid}))
            if len(self.team) != 1:
                raise Exception("Could not find %s's team" % self.pid)
            else:
                self.team = self.team[0]
            self.player_team = self.team["_id"]
            self.ts = self.upcoming_game['time']
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            self.opp_team = team2_id if (self.player_team == team1_id) else team2_id

        # get all teams and locations
        results = team_collection.find({})
        self.all_teams = set()
        self.all_locations = set()
        for r in results:
            self.all_teams.add(r['_id'])
            self.all_locations.add(r['location'])
        self.all_teams = list(self.all_teams)
        self.all_locations = list(self.all_locations)

    def timestamp(self):
        return self.ts

    def runEncoderFeatures(self):
        """
        DENOISING FEATURES:
            - home/away espn team features
            - home/away running avg team features
            - player running avg features
            - team intensity features (home/away winrate/streak)
            - indv intensity features (awaysInLast10/MPinlast10) (lower to 6 games)
            - location
            - back to back
            - opposing team
            - previous matchup stats (point total, team pts, record)
            - previous matchup player running avg stats
            
        DENOISING DIFF FEATURES
            - team win percentage difference
            - previous matchup team stat difference
            - espn team stat difference
            - running avg team stat difference

        TODO:
        - team positional height differences (by position, by depth)
        - defensive stats of the opposing team's best defensive lineup
        - espn player tracking stats for player
        - difference between espn/season stats and recent averages (=> recent trends)
        - difference between espn/season stats and previous matchup player and team averages (=> previous matchup trends)

        REMOVE ALL DIFF STATS
        JUST ENCODE OWN STATS AND OPP STATS
        INCLUDE PLAYER STATS FOR OWN PLAYER'S MOST LIKELY LINEUP AND OPPOSITIONS BEST DEFENSIVE LINEUP

        """
        fns = [self.runOppositionPlayerFeatures, self.runIntensityFeatures, self.runTeamFeatures, self.runPlayerFeatures, self.runMatchupFeatures, self.runPhysicalFeatures]
        cat_labels_all = []
        cat_features_all = []
        cont_labels_all = []
        cont_features_all = []
        cat_feat_splits_all = []
        for fn in fns:
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = fn()
            cat_labels_all += cat_labels
            cat_features_all += cat_features
            cont_labels_all += cont_labels
            cont_features_all += cont_features
            cat_feat_splits_all += cat_feat_splits

        return cat_labels_all, cat_features_all, cont_labels_all, cont_features_all, cat_feat_splits_all

    def runMatchupFeatures(self):
        own_avg, opp_avg, diff, pid = self.matchupFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []

        for tkey in self.TEAM_STATS:
            value = own_avg[tkey]
            label = "%s_%s" % ('matchup_own', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        for tkey in self.TEAM_STATS:
            value = opp_avg[tkey]
            label = "%s_%s" % ('matchup_opp', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        for tkey in self.TEAM_STATS:
            value = diff[tkey]
            label = "%s_%s" % ('matchup_diff', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        for tkey in self.PLAYER_STATS:
            value = pid[tkey]
            label = "%s_%s" % ('matchup_pid', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runIntensityFeatures(self):
        features = self.intensityFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []

        # opp team earned
        for tkey in self.INTENSITY_STATS:
            value = features[tkey]
            label = "%s_%s" % ('intensity', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runPhysicalFeatures(self):
        opp_team, opp_loc = self.oppositionLocationFeatures()
        bb = self.backTobackFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []

        # opp team one hot
        opp_team_keys = sorted(opp_team.keys())
        cat_features.append([opp_team[k] for k in opp_team_keys])
        cat_labels.append(opp_team_keys)

        # game location one hot
        loc_keys = sorted(opp_loc.keys())
        cat_features.append([opp_loc[k] for k in loc_keys])
        cat_labels.append(loc_keys)
        
        # back to back features
        bb_keys = sorted(bb.keys())
        cat_features.append([bb[k] for k in bb_keys])
        cat_labels.append(bb_keys)
        
        # flatten categorical features
        cat_features = [item for sublist in cat_features for item in sublist]
        cat_labels = [item for sublist in cat_labels for item in sublist]

        # features splits
        cat_feat_splits = [len(opp_team.keys()), len(opp_loc.keys()), len(bb.keys())]

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runTeamFeatures(self):
        own_avgs, opp_avgs, diffs = self.teamStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []

        for tkey in self.TEAM_STATS + self.ESPN_TEAM_STATS:
            value = own_avgs[tkey]
            label = "%s_%s" % ('own_team', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        for tkey in self.TEAM_STATS + self.ESPN_TEAM_STATS:
            value = opp_avgs[tkey]
            label = "%s_%s" % ('opp_team', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        for tkey in self.TEAM_STATS + self.ESPN_TEAM_STATS:
            value = diffs[tkey]
            label = "%s_%s" % ('team_diff', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runOppositionPlayerFeatures(self):
        op_allowed = self.opposingPlayerAllowedStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []
        # opp player stats
        for pkey in self.PLAYER_STATS:
            value = op_allowed[pkey]
            label = "%s_%s" % ('op_allowed', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runPlayerFeatures(self):
        avgs, facing_avgs, diff = self.playerStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []
        # player stats
        for pkey in self.PLAYER_STATS:
            value = avgs[pkey]
            label = "%s_%s" % ('pstat', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # facing stats
        for pkey in self.PLAYER_STATS:
            value = facing_avgs[pkey]
            label = "%s_%s" % ('fstat', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # diff stats
        for pkey in self.PLAYER_STATS:
            value = diff[pkey]
            label = "%s_%s" % ('pf_diff', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []
        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)


    def getY(self, key):
        if self.player_game_info:
            return self.player_game_info[key]
        else:
            raise Exception("Game is upcoming")

    def matchupFeatures(self):
        '''
        FEATURES:
            previous matchup stats (point total, team pts, record)
            previous matchup player running avg stats
        DIFF FEATURES:
            previous matchup team stat difference
            espn tracking and previous matchup player avgs diff
        '''
        # find previous matchups between the two teams
        prev_matches = list(game_collection.find({"teams": [self.player_team, self.opp_team]}))
        if prev_matches:
            prev_matches_own = [team_game_collection.find_one({"team_id": self.player_team, "game_id": i['_id']}) for i in prev_matches]
            prev_matches_opp = [team_game_collection.find_one({"team_id": self.opp_team, "game_id": i['_id']}) for i in prev_matches]
            prev_matches_pid = []
            for g in prev_matches:
                prev_game = player_game_collection.find_one({"player_id": self.pid, "game_id": g["_id"]})
                if prev_game and prev_game['MP'] > 0.0:
                    prev_matches_pid.append(prev_game)
            if prev_matches_pid:
                prev_matches_pid = self.convertPlayerGameStatsToPer48(prev_matches_pid)
                prev_matches_pid_avg = self.averageStats(prev_matches_pid, self.PLAYER_STATS)
            else:
                prev_matches_pid_avg = {k:np.nan for k in self.PLAYER_STATS}
            prev_matches_own_avg = self.averageStats(prev_matches_own, self.TEAM_STATS)
            prev_matches_opp_avg = self.averageStats(prev_matches_opp, self.TEAM_STATS)
            prev_match_diff = {k: v - prev_matches_opp_avg[k] for k,v in prev_matches_own_avg.items()}
        else:
            prev_matches_opp_avg = {k: np.nan for k in self.TEAM_STATS}
            prev_matches_own_avg = {k: np.nan for k in self.TEAM_STATS}
            prev_match_diff = {k: np.nan for k in self.TEAM_STATS}
            prev_matches_pid_avg = {k: np.nan for k in self.PLAYER_STATS}

        return (prev_matches_own_avg, prev_matches_opp_avg, prev_match_diff, prev_matches_pid_avg)

    def intensityFeatures(self):
        '''
        FEATURES:
            team intensity features (home/away winrate/streak)
            indv intensity features (awaysInLast/MPinlast) 5 games
        DIFF FEATURES:
            win percentage difference
        '''
        own_team = self.player_team
        if self.target_game_info:
            game_time = self.team_game_info['game_time']
            if self.target_game_info['home_id'] == own_team:
                opp_team = self.target_game_info['away_id']
            else:
                opp_team = self.target_game_info['home_id']
        else:
            game_time = self.upcoming_game['time']
            # find the team ids by team name
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            opp_team = team2_id if (self.player_team == team1_id) else team2_id
        own_query = {"teams": own_team, "time": {"$lt": game_time}}
        opp_query = {"teams": opp_team, "time": {"$lt": game_time}}
        own_team_games = [_ for _ in game_collection.find(own_query, sort=[("time",-1)], limit=10)]
        opp_team_games = [_ for _ in game_collection.find(opp_query, sort=[("time",-1)], limit=10)]
        own_stats = [team_game_collection.find_one({"team_id": own_team, "game_id": _["_id"]}) for _ in own_team_games]
        opp_stats = [team_game_collection.find_one({"team_id": opp_team, "game_id": _["_id"]}) for _ in opp_team_games]
        sixdays_ago = game_time - timedelta(6.5)
        player_stats = player_game_collection.find({"player_id": self.pid, "game_time" : {"$lt": game_time, "$gt": sixdays_ago}})
        mp_in_6 = sum([float(_.get('MP',0.0)) for _ in player_stats])
        own_most_recent = own_team_games[0] if len(own_team_games)>0 else None
        opp_most_recent = opp_team_games[0] if len(opp_team_games)>0 else None

        # calculate number of aways in last 6
        own_aways = len([_ for _ in own_stats[:6] if _.get('location') == 'Away'])
        opp_aways = len([_ for _ in opp_stats[:6] if _.get('location') == 'Away'])

        # parse records and streaks
        if own_most_recent is None:
            own_record = None
        elif own_team == own_most_recent['home_id']:
            own_record = own_most_recent['home_record']
            own_streak = own_most_recent['home_streak']
        else:
            own_record = own_most_recent['away_record']
            own_streak = own_most_recent['away_streak']
        if opp_most_recent is None:
            opp_record = None
        elif opp_team == opp_most_recent['home_id']:
            opp_record = opp_most_recent['home_record']
            opp_streak = opp_most_recent['home_streak']
        else:
            opp_record = opp_most_recent['away_record']
            opp_streak = opp_most_recent['away_streak']

        if own_record is None:
            own_winning_percent = 1.0
            own_streak = 0.0
        else:
            own_record_parts = own_record.split('-')
            own_wins = int(own_record_parts[0].strip())
            own_losses = int(own_record_parts[1].strip())
            own_winning_percent = own_wins / float(own_losses + own_wins)
            if 'Lost' in own_streak:
                own_streak = -int(own_streak.replace('Lost ','').strip())
            else:
                own_streak = int(own_streak.replace('Won ','').strip())
        if opp_record is None:
            opp_winning_percent = 1.0
            opp_streak = 0.0
        else:
            opp_record_parts = opp_record.split('-')
            opp_wins = int(opp_record_parts[0].strip())
            opp_losses = int(opp_record_parts[1].strip())
            opp_winning_percent = opp_wins / float(opp_losses + opp_wins)
            if 'Lost' in opp_streak:
                opp_streak = -int(opp_streak.replace('Lost ','').strip())
            else:
                opp_streak = int(opp_streak.replace('Won ','').strip())

        # load feature dictionary
        data = {'own_win_rate': own_winning_percent,
                'opp_win_rate': opp_winning_percent,
                'own_streak': own_streak,
                'opp_streak': opp_streak,
                'win_rate_diff': own_winning_percent-opp_winning_percent,
                'own_aways_in_6': own_aways,
                'opp_aways_in_6': opp_aways,
                'mp_in_6': mp_in_6
                }
        return data


    def oppositionLocationFeatures(self):
        '''
        FEATURES:
            location
            opposing team
        DIFF FEATURES:
        '''
        if self.target_game_info:
            home_team = team_collection.find_one({"_id":self.target_game_info['home_id']})
        else:
            home_team = team_collection.find_one({"name":self.upcoming_game['home_team_name']})

        if not home_team:
            raise Exception("Could not find HOME team")

        game_location = home_team['location']
        opp_teams_one_hot = {x:-1 for x in self.all_teams}
        location_one_hot = {x:-1 for x in self.all_locations}

        try:
            opp_teams_one_hot[self.opp_team] = 1
            location_one_hot[game_location] = 1
        except ValueError as e:
            raise Exception("Team or location not found in DB")

        return (opp_teams_one_hot, location_one_hot)

    def backTobackFeatures(self):
        """
        FEATURES:
            back to back features
        DIFF FEATURES:
        """
        b2b = -1
        two_three = -1
        three_four = -1
        four_five = -1
        if self.team_game_info:
            # find the last 3 games and check if they are within the day ranges
            game_time = self.team_game_info['game_time']
            last_3_games = team_game_collection.find({"team_id": self.player_team,
                                                     "game_time" : {"$lt" : game_time}},
                                                     sort=[("game_time",-1)], 
                                                     limit=3)
            last_game_times = [x['game_time'] for x in last_3_games]
        else:
            game_time = self.upcoming_game['time']
            last_3_games = team_game_collection.find({"team_id": self.player_team,
                                                     "game_time" : {"$lt" : game_time}},
                                                     sort=[("game_time",-1)], 
                                                     limit=3)
            last_3_games_future = future_collection.find({"teams": self.player_team,
                                                         "time" : {"$lt" : game_time}}, 
                                                         sort=[("time",-1)],
                                                         limit=3)
            last_game_times = [x['game_time'] for x in last_3_games]
            for g in last_3_games_future:
                last_game_times.append(g['time'])
        game_day = datetime(year=game_time.year,month=game_time.month,day=game_time.day)
        last_game_days = [datetime(year=x.year,month=x.month,day=x.day) for x in last_game_times]
        last_game_deltas = [(game_day - x).days for x in last_game_days]
        game_index = [0,0,0,0]
        for d in last_game_deltas:
            if d < 5:
                game_index[d-1] = 1
        if sum(game_index[:1]) == 1:
            b2b = 1
        if sum(game_index[:2]) == 1:
            two_three = 1
        if sum(game_index[:3]) == 2:
            three_four = 1
        if sum(game_index[:4]) == 3:
            four_five = 1
        return {"b2b":b2b, "2of3": two_three, "3of4": three_four, "4of5": four_five}


    def averageStats(self, stats, allowed_keys):
        trajectories = {k:[] for k in allowed_keys}
        for stat in stats:
            for k in allowed_keys:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            values = [x for x in v if x is not None]
            if len(values) > 0:
                trajectories[k] = np.mean(values)
            else:
                trajectories[k] = np.nan
        return trajectories

    def convertPlayerGameStatsToPer48(self, stats):
        new_stats = []
        for s in stats:
            mp = float(s['MP'])
            if mp == 0.0:
                new_stats.append(s)
            else:
                ratio = 48.0 / mp
                for k in self.PER48_STATS:
                    s[k] = s[k] * ratio
                new_stats.append(s)
        return new_stats

    def playerStatFeatures(self):
        """
        FEATURES:
            player running avg players
            matchup player running avg features
            espn tracking stats
        DIFF FEATURES:
            matchup player running avg stat difference
            espn tracking/recent avg difference
        """
        # grab the most recent 3-10 games for the player leading up to the target game
        results = player_game_collection.find({"player_id": self.pid,
                                               "game_time": {"$lt": self.ts}}, 
                                              sort=[("game_time",-1)], 
                                              limit=10)
        results = list(results)

        if len(results) < 3:
            raise Exception("Could not find enough prior games for player %s" % self.pid)

        results = self.convertPlayerGameStatsToPer48(results)
        avgs = self.averageStats(results, self.PLAYER_STATS)
        
        # find closest espn player tracking stat row
        tracking = list(espn_player_stat_collection.find({"player_id": self.pid}))
        now = datetime.now()
        sorted_tracking = sorted(tracking, key=lambda x: abs(now-x['time']))
        recent_tracking = sorted_tracking[0]

        # find the diff for the relevant stats
        diff = {}

        return (avgs, facing_avgs, diff)

    def facingPlayerFeatures(self):
        """
        DEPRECATEDa

        FEATURES:
            matchup player running avg features
        DIFF FEATURES:
        """
        query = {"team_id": self.opp_team, "game_time": {"$lt": self.ts}}
        results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
        results = list(results)

        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % query)

        # take the first position player plays
        position = self.player_info["position"][0]
        all_player_stats = defaultdict(list)
        mins_played = defaultdict(list)
        # find player on opp team that plays the most minutes in that position
        for r in results:
            player_game_rows = player_game_collection.find({"game_id" : r['game_id'], "player_team": self.opp_team})
            for game_row in player_game_rows:
                pid = game_row['player_id']
                # check player position
                player_row = player_collection.find_one({"_id" : pid})
                if not player_row:
                    print "Could not find %s for stats" % pid
                    continue
                if position not in player_row['position']:
                    continue
                all_player_stats[pid].append(game_row)
                mins_played[pid].append(game_row['MP'])
        if len(mins_played) == 0:
            raise Exception("No facing %s players found for team %s at %s" % (position, self.opp_team, self.ts))
        sorted_pid = sorted([(k, sum(v)) for k,v in mins_played.items()], key=lambda x: x[1], reverse=True)
        used_pid = sorted_pid[0][0]
        used_stats = all_player_stats[used_pid]
        used_stats = self.convertPlayerGameStatsToPer48(used_stats)

        avgs = self.averageStats(used_stats, self.PLAYER_STATS)
        return avgs

    def opposingPlayerAllowedStatFeatures(self):
        '''
        defensive stats of the opposing team's best defensive lineup

        exclude players that aren't playing
        '''
        pass

    def teamStatFeatures(self):
        '''
        FEATURES:
            home/away espn team features
            home/away running avg team features
        DIFF FEATURES:
            espn team stat difference
            running avg team stat difference
        '''
        # for opposing team
        query = {"team_id": self.opp_team, "game_time": {"$lt": self.ts}}
        results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
        results = list(results)
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % query)

        opp_avgs = self.averageStats(results, self.TEAM_STATS)

        # for  own team
        query = {"team_id": self.player_team, "game_time": {"$lt": self.ts}}
        results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
        results = list(results)
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % query)

        own_avgs = self.averageStats(results, self.TEAM_STATS)

        # get espn team stats, look backward
        espnstats = list(espn_stat_collection.find({"time" : {"$lt" : self.ts}}, sort= [("time",-1)],limit=1))
        if len(espnstats) == 0:
            # look forward
            espnstats = list(espn_stat_collection.find({"time" : {"$gt" : self.ts}}, sort= [("time",1)], limit=1))
        espnstat_opp = espnstats[0]['stats'][self.opp_team]
        espnstat_self = espnstats[0]['stats'][self.player_team]

        for k in espnstat_self.keys():
            new_k = "ESPN_%s" % k
            opp_avgs[new_k] = espnstat_opp[k]
            own_avgs[new_k] = espnstat_self[k]

        # get diff
        diffs = {k: v - opp_avgs[k] for k,v in own_avgs.iteritems()}

        return (own_avgs, opp_avgs, diffs)

'''
#UPCOMING GAME
{'home_team_name': u'Utah Jazz', 'away_team_name': u'Sacramento Kings', 
'time': datetime.datetime(2015, 2, 7, 21, 0)}

#GAME
{ "_id" : "201502270CHI", 
"away_pts" : 89, 
"inactive" : [ "bennean01", "muhamsh01", "gasolpa01", "rosede01" ], 
"time of game" : "2:12", 
"home_pts" : 96, 
"away_id" : "MIN", 
"attendance" : 21635, 
"home_record" : "37-22", 
"url" : "http://www.basketball-reference.com/boxscores/201502270CHI.html", 
"teams" : [ "MIN", "CHI" ], 
"home_id" : "CHI", 
"officials" : [ "brothto99r", "collide99r", "workmha01r" ], 
"home_streak" : "Won 1", 
"time" : ISODate("2015-02-27T20:00:00Z"), 
"away_streak" : "Lost 1", 
"away_record" : "13-44", 
"location" : "United Center, Chicago, Illinois" }

#PLAYER
{ "_id" : "allenra02", "weight" : 205, 
"url" : "http://www.basketball-reference.com/players/a/allenra02.html", 
"experience" : 18, "height" : 77, "born" : ISODate("1975-07-20T00:00:00Z"), 
"full_name" : "Walter Ray Allen", "position" : [ "sg" ], "shoots" : "R", 
"nba debut" : ISODate("1996-11-01T00:00:00Z") }

#TEAM
{ "_id" : "ATL", "url" : "http://www.basketball-reference.com/teams/ATL/", 
"players" : [ "horfoal01", "jenkijo01", "macksh01", "scottmi01", 
"schrode01", "bazemke01", "sefolth01", "millspa01", "korveky01", "teaguje01", 
"brandel01", "anticpe01", "carrode01", "muscami01", "paynead01" ], 
"location" : "Atlanta, Georgia", "name" : "Atlanta Hawks" }

#PLAYER_GAME
{ "_id" : ObjectId("54d538048c24bd43324351d3"), "STL%" : 1.4, "FT" : 3, "3P" : 0, 
"TOV" : 4, "FG" : 6, "player_id" : "walljo01", "3PA" : 1, "DRB" : 3, "ORB%" : 0, 
"BLK%" : 0, "AST" : 13, "game_time" : ISODate("2015-02-05T19:00:00Z"), "FT%" : 0.75, 
"3PAr" : 0.077, "PF" : 1, "PTS" : 15, "FGA" : 13, "FG%" : 0.462, "STL" : 1, "TRB" : 3, 
"TOV%" : 21.3, "AST%" : 59.5, "FTA" : 4, "eFG%" : 0.462, "BLK" : 0, "game_id" : "201502050CHO", 
"FTr" : 0.308, "+/-" : 9, "player_team" : "WAS", "USG%" : 22.8, "DRB%" : 9.4, "TS%" : 0.508, 
"MP" : 38.18333333333333, "DRtg" : 106, "ORtg" : 101, "TRB%" : 4.3, "ORB" : 0, "3P%" : 0 }

#TEAM_GAME
{ "_id" : ObjectId("54d538048c24bd43324351d1"), "location" : "Away", 
"game_time" : ISODate("2015-02-05T19:00:00Z"), "TOV%" : 11.6, "team_id" : "WAS", 
"1" : 30, "Pace" : 89.9, "3" : 20, "2" : 24, "4" : 13, "ORtg" : 96.8, 
"game_id" : "201502050CHO", "FT/FGA" : 0.143, "ORB%" : 19.1, 
"eFG%" : 0.446, "T" : 87 }

'''

if __name__=="__main__":
    pass



