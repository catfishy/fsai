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


def findAllTrainingGames(pid):
    '''
    TODO: FILTER games by player's current team????? (or just general teammate context)
    '''

    # find all training games
    playergames = list(player_game_collection.find({"player_id": pid}, sort=[("game_time",-1)]))
    # drop the first 3 games
    playergames = playergames[:-3]
    # drop games where mp == 0
    valid_games = [g for g in playergames if g['MP'] != 0.0]
    return valid_games


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
    TEAM_STATS = ["TOV%", "Pace", "ORtg", "FT/FGA", "ORB%", "eFG%",  "T"]

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

    def runOppositionEarnedFeatures(self):
        ot_earned, ot_allowed = self.opposingTeamStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []
        # opp team earned
        for tkey in self.TEAM_STATS:
            value = ot_earned[tkey]
            label = "%s_%s" % ('ot_earned', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []

        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def runOppositionAllowedFeatures(self):
        ot_earned, ot_allowed = self.opposingTeamStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []
        # opp team allowed
        for tkey in self.TEAM_STATS:
            value = ot_allowed[tkey]
            label = "%s_%s" % ('ot_allowed', tkey)
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
        pstat = self.playerStatFeatures()
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []
        # player stats
        for pkey in self.PLAYER_STATS:
            value = pstat[pkey]
            label = "%s_%s" % ('pstat', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # features splits
        cat_feat_splits = []
        return (cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits)

    def run(self):
        pstat = self.playerStatFeatures()
        op_allowed = self.opposingPlayerAllowedStatFeatures()
        ot_earned, ot_allowed = self.opposingTeamStatFeatures()
        opp_team, opp_loc = self.oppositionLocationFeatures()
        bb = self.backTobackFeatures()
        
        # order them
        cont_features = []
        cont_labels = []
        cat_features = []
        cat_labels = []

        # player stats
        for pkey in self.PLAYER_STATS:
            value = pstat[pkey]
            label = "%s_%s" % ('pstat', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # opp player stats
        for pkey in self.PLAYER_STATS:
            value = op_allowed[pkey]
            label = "%s_%s" % ('op_allowed', pkey)
            cont_features.append(value)
            cont_labels.append(label)

        # opp team earned
        for tkey in self.TEAM_STATS:
            value = ot_earned[tkey]
            label = "%s_%s" % ('ot_earned', tkey)
            cont_features.append(value)
            cont_labels.append(label)

        # opp team allowed
        for tkey in self.TEAM_STATS:
            value = ot_allowed[tkey]
            label = "%s_%s" % ('ot_allowed', tkey)
            cont_features.append(value)
            cont_labels.append(label)

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

    def getY(self, key):
        if self.player_game_info:
            return self.player_game_info[key]
        else:
            raise Exception("Game is upcoming")

    def oppositionLocationFeatures(self):
        if self.target_game_info:
            if self.player_team == self.target_game_info['team1_id']:
                opposing_team = self.target_game_info['team2_id']
            else:
                opposing_team = self.target_game_info['team1_id']
            home_team = team_collection.find_one({"_id":self.target_game_info['team2_id']})
            if not home_team:
                raise Exception({"_id":self.target_game_info['team2_id']})
            game_location = home_team['location']
        else:
            # find the team ids by team name
            team1 = team_collection.find_one({"name":self.upcoming_game['home_team_name']})
            team1_id = team1["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            if self.player_team == team1_id:
                opposing_team = team2_id
            else:
                opposing_team = team1_id
            game_location = team1['location']

        opp_teams_one_hot = {x:-1 for x in self.all_teams}
        location_one_hot = {x:-1 for x in self.all_locations}

        try:
            opp_teams_one_hot[opposing_team] = 1
            location_one_hot[game_location] = 1
        except ValueError as e:
            raise Exception("Team or location not found in DB")

        return (opp_teams_one_hot, location_one_hot)

    def backTobackFeatures(self):
        """
        NBA never plays 3 b2b games
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

    def playerStatFeatures(self):
        if not self.target_game_id:
            # grab most recent 3-10 games for the player
            results = player_game_collection.find({"player_id":self.pid}, 
                                                  sort=[("game_time",-1)], 
                                                  limit=10)
            results = list(results)
        else:
            # grab the most recent 3-10 games for the player leading up to the target game
            target_game_time = self.target_game_info["time"]
            results = player_game_collection.find({"player_id": self.pid,
                                                   "game_time": {"$lt": target_game_time}}, 
                                                  sort=[("game_time",-1)], 
                                                  limit=10)
            results = list(results)
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for player %s" % self.pid)
        trajectories = {k:[] for k in self.PLAYER_STATS}
        for stat in results:
            for k in self.PLAYER_STATS:
                trajectories[k].append(stat.get(k))
        # average out the stats
        # TODO: WEIGH MORE RECENT EVERYWHERE
        for k,v in trajectories.iteritems():
            trajectories[k] = np.mean(np.mean([x for x in v if x is not None]))
        return trajectories

    def opposingPlayerAllowedStatFeatures(self):
        # for the games the opposing team has played, 
        # find the player opposing the opposing team who played the same position
        # and average those stats
        if not self.target_game_id:
            # find the team ids by team name
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            if self.player_team == team1_id:
                opp_team = team2_id
            else:
                opp_team = team1_id
            # grab most recent 3-10 games for the team
            query = {"team_id": opp_team}
            results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
            results = list(results)
        else:
            if self.player_team == self.target_game_info['team1_id']:
                opp_team = self.target_game_info['team2_id']
            else:
                opp_team = self.target_game_info['team1_id']
            # grab the most recent 3-10 games for the player leading up to the target game
            target_game_time = self.target_game_info["time"]
            query = {"team_id": opp_team, "game_time": {"$lt": target_game_time}}
            results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
            results = list(results)
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % query)
        # use game ids for allowed stats
        game_ids = [x['game_id'] for x in results]
        # calculate allowed stats
        allowed_team_stats = []
        for gid in game_ids:
            result = team_game_collection.find_one({"game_id": gid, "team_id": {'$ne': opp_team}})
            allowed_team_stats.append(result)
        # find the player stats for these games for the same position
        position = self.player_info["position"][0] # take the first position player plays
        player_stats = defaultdict(list)
        for team_stat in allowed_team_stats:
            players_in_game = list(player_game_collection.find({"game_id": team_stat['game_id'], 
                                                                "player_team": team_stat['team_id']}))
            for pstat in players_in_game:
                player_id = pstat['player_id']
                player_row = player_collection.find_one({"_id": player_id})
                if not player_row:
                    print "Could not find player %s" % player_id
                    continue
                if position in player_row['position']:
                    player_stats[team_stat['game_id']].append(pstat)
        used_stats = []
        for k,v in player_stats.iteritems():
            if len(v) > 1:
                # choose one with the most minutes
                mp = [x['MP'] for x in v]
                max_index = mp.index(max(mp))
                used_stats.append(v[max_index])
            else:
                used_stats.append(v[0])
        trajectories = {k:[] for k in self.PLAYER_STATS}
        for stat in used_stats:
            for k in self.PLAYER_STATS:
                trajectories[k].append(stat.get(k))
        # average out the stats
        for k,v in trajectories.iteritems():
            trajectories[k] = np.mean([x for x in v if x is not None])
        return trajectories

    def opposingTeamStatFeatures(self):
        if not self.target_game_id:
            # find the team ids by team name
            team1_id = team_collection.find_one({"name":self.upcoming_game['home_team_name']})["_id"]
            team2_id = team_collection.find_one({"name":self.upcoming_game['away_team_name']})["_id"]
            if self.player_team == team1_id:
                opp_team = team2_id
            else:
                opp_team = team1_id
            # grab most recent 3-10 games for the team
            query = {"team_id": opp_team}
            results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
            results = list(results)
        else:
            if self.player_team == self.target_game_info['team1_id']:
                opp_team = self.target_game_info['team2_id']
            else:
                opp_team = self.target_game_info['team1_id']
            # grab the most recent 3-10 games for the team leading up to the target game
            target_game_time = self.target_game_info["time"]
            query = {"team_id": opp_team, "game_time": {"$lt": target_game_time}}
            results = team_game_collection.find(query, sort=[("game_time",-1)], limit=10)
            results = list(results)
        if len(results) < 3:
            raise Exception("Could not find more than 3 prior games for team: %s" % query)

        # use game ids for allowed stats
        game_ids = [x['game_id'] for x in results]

        # calculate earned stats
        earned_trajectories = {k:[] for k in self.TEAM_STATS}
        for stat in results:
            for k in self.TEAM_STATS:
                earned_trajectories[k].append(float(stat[k]))
        # average out the stats
        for k,v in earned_trajectories.iteritems():
            earned_trajectories[k] = np.mean(v)

        # calculate allowed stats
        allowed_team_stats = []
        for gid in game_ids:
            result = team_game_collection.find_one({"game_id": gid, "team_id": {'$ne': opp_team}})
            allowed_team_stats.append(result)
        allowed_trajectories = {k:[] for k in self.TEAM_STATS}
        for stat in allowed_team_stats:
            for k in self.TEAM_STATS:
                allowed_trajectories[k].append(float(stat[k]))
        # average out the stats
        for k,v in allowed_trajectories.iteritems():
            allowed_trajectories[k] = np.mean(v)

        return (earned_trajectories,allowed_trajectories)

'''
#UPCOMING GAME
{'home_team_name': u'Utah Jazz', 'away_team_name': u'Sacramento Kings', 
'time': datetime.datetime(2015, 2, 7, 21, 0)}

#GAME
{ "_id" : "201502050CHO", "team1_pts" : 87, "team1_streak" : "Lost 5", 
"inactive" : [ "goodedr01", "biyombi01", "walkeke02" ], "team1_record" : "31-20", 
"time of game" : "2:15", "attendance" : 17019, "team2_pts" : 94, 
"url" : "http://www.basketball-reference.com/boxscores/201502050CHO.html", 
"team2_record" : "22-27", "officials" : [ "richale99r", "roeel99r", "wrighse99r" ], 
"team2_id" : "CHO", "time" : ISODate("2015-02-05T19:00:00Z"), "team1_id" : "WAS", 
"team2_streak" : "Won 3", "location" : "Time Warner Cable Arena, Charlotte, North Carolina" }

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
    pid = "curryst01"
    print pid

    playergames = findAllTrainingGames(pid)
    X = []
    Y = []
    for g in playergames:
        gid = str(g['game_id'])
        fext = featureExtractor(pid, target_game=gid)
        cat_labels, cat_features, cont_labels, cont_features = fext.run()
        y = fext.getY('PTS')
        X.append((cont_features,cat_features))
        Y.append(y)



