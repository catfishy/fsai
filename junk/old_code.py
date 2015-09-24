# DEPRECATED
def convertNBAPlayerID(player_id):
    # try looking in db
    db_row = player_collection.find_one({'nba_id': player_id})
    if db_row is not None and db_row.get('nba_position') is not None:
        return db_row['_id']
    # match the player up to a BR ID
    nba_info = crawlNBAPlayerInfo(player_id)
    birthstamp = nba_info['BIRTHDATE'].values[0]
    player_name = nba_info['DISPLAY_FIRST_LAST'].values[0]
    player_position = nba_info['POSITION'].values[0]
    print player_name
    birthstamp_year = birthstamp.split('T')[0].strip()
    birthstamp_datetime = datetime.strptime(birthstamp_year, "%Y-%m-%d")
    player_dict = translatePlayerNames([player_name], player_birthdays=[birthstamp_datetime])[player_name]
    if len(player_dict) == 0: # try to crawl for it
        new_crawl = crawlBRPlayer(player_name)
        if new_crawl:
            # re-search
            player_dict = translatePlayerNames([player_name], player_birthdays=[birthstamp_datetime])[player_name]
    if len(player_dict) == 0:
        print "No match for %s, %s" % (player_name, birthstamp_datetime)
        return None
    if len(player_dict) > 1:
        print 'SOLVE AMBIGUITY: %s' % player_dict
        return None
    player_row = player_dict[0]
    br_player_id = player_row['_id']
    update_dict = {'nba_id': player_id, 'nba_position': player_position}
    nba_conn.updateDocument(player_collection, br_player_id, update_dict, upsert=False)
    return br_player_id



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