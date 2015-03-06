'''
maintaining db and running jobs for upcoming games that are bet on
maintaining a db with roster, salary, payroll, scoring for all active games
matching rosters up with db players
training models + making predictions for all stats X players
converting predictions to projected points
make best roster with salary, payroll, and projected points

get automatic refresher on game sites

TODO: modularize fanduel analysis and extend to draftkings

'''
import copy
import itertools
from collections import defaultdict
from datetime import datetime
import time
import numpy as np
import traceback
import multiprocessing as mp
import Queue
import math

import matplotlib.pyplot as plt

from statsETL.db.mongolib import *
from analysis.bball.train import *
from analysis.bball.playerAnalysis import featureExtractor, findAllTrainingGames
from analysis.bball.rosterAnalysis import FanDuelOptimalRoster
from analysis.util.kimono import *

FANDUEL_PT_VALUES = {'TOV': -1.0, 
                     'AST': 1.5,
                     'STL': 2.0,
                     'TRB': 1.2,
                     'BLK': 2.0,
                     'PTS': 1.0}

TO_BR_TEAM_TRANSLATIONS = {'SA': 'SAS',
                           'GS': 'GSW',
                           'NO': 'NOP',
                           'CHA': 'CHO',
                           'NY': 'NYK',
                           'BKN': 'BRK'
                          }

Y_KEYS = ['PTS','TRB','AST','TOV','STL','BLK','MP','USG%']

def modelPlayersInUpcomingGames(logger, days_ahead=3, poolsize=4):
    upper_limit = datetime.now() + timedelta(days_ahead)
    upcoming_games = future_collection.find({"time": {"$gt" : datetime.now()}}, sort=[("time",1)])
    valid_games = [g for g in upcoming_games if g['time'] <= upper_limit]
    logger.info("Found %s upcoming games before %s" % (len(valid_games),upper_limit))
    for matched_game in valid_games:
        logger.info("Modeling game: %s" % matched_game)
        avai_players = availablePlayers(matched_game)
        home_name = matched_game['home_team_name']
        away_name = matched_game['away_team_name']
        home_players = avai_players[home_name]
        away_players = avai_players[away_name]
        all_players = home_players + away_players
        logger.info("Players: %s" % all_players)
        process_inputs = []
        invalid_players = []
        stat_projections = {}
        for pid in all_players:
            process_inputs.append((pid, matched_game))

        # in parallel, train and project players that need it
        args = iter(process_inputs)
        pool = mp.Pool(poolsize)
        pool_results = pool.imap_unordered(trainAndProjectPlayer, args)
        pool.close()
        pool.join()
        for pid,pid_results in pool_results:
            if pid_results is None:
                invalid_players.append(pid)
            else:
                stat_projections[pid] = pid_results        
        logger.info("%s: %s players projected, %s invalid" % (matched_game['_id'], len(stat_projections), len(invalid_players)))
    return stat_projections

def analyzeFanDuelGame(game_id, logger, crawl=True):
    """
    create new game row in db
    create kimono api, continuously update scraped values (LOG)
    train models or get already trained ones (LOG to ?dynamo?)
    continuously update optimal roster + save to db (LOG UPDATES)
    """
    new_url = createFanDuelDraftURL(game_id)

    logger.info("Fanduel URL: %s" % new_url)

    kimono_info = fanDuelNBADraftAPIContent(new_url, crawl=crawl)
    kimono_info = mergeFanduelDB(game_id, kimono_info)
    
    # build players by game lookup
    players_by_game = defaultdict(list)
    for pid, game in kimono_info['player_games'].iteritems():
        players_by_game[game].append(pid)
    valid_game_ids = set(kimono_info['player_games'].values())

    # kimono invalid players, adjusted by manual in/validation
    invalid_players = list(set(kimono_info['invalids'] + kimono_info['manual_invalids']))
    for pid in kimono_info['manual_valids']:
        if pid in invalid_players:
            invalid_players.pop(pid)

    logger.info("Fanduel Invalids: %s" % invalid_players)
    logger.info("Fanduel GTD: %s" % kimono_info['gtds'])

    # match up game ids with upcoming games
    upcoming_by_id = {} 
    player_teams = {}
    for gid in valid_game_ids:
        teams = gid.split('@')
        away = teams[0]
        home = teams[1]
        matched_game = matchUpcomingGame(home, away)

        logger.info("Matched %s with %s" % (gid, matched_game))

        upcoming_by_id[gid] = matched_game

        # find players in game
        game_pids = players_by_game[gid]
        home_name = matched_game['home_team_name']
        away_name = matched_game['away_team_name']
        avai_players = availablePlayers(matched_game)
        home_players = avai_players[home_name]
        away_players = avai_players[away_name]

        for p in game_pids:
            if p in home_players:
                player_teams[p] = home_name
            elif p in away_players:
                player_teams[p] = away_name
            else:
                logger.info("%s not on %s or %s rosters" % (p,home_name,away_name))
                invalid_players.append(p)
    kimono_info['player_teams'] = player_teams

    logger.info("Invalids from Crawl: %s, %s" % (len(invalid_players), invalid_players))

    # train/project for players if necessary
    process_inputs = []
    to_train_pids = []
    stat_projections = {}
    for pid in kimono_info['players']:
        # skip invalid players
        if pid in invalid_players:
            continue
        # get the matched game
        gid = kimono_info['player_games'][pid]
        matched_game = upcoming_by_id[gid]
        # look for previous trainings
        prev_proj = getPreviousProjection(pid, matched_game)
        if not prev_proj:
            process_inputs.append((pid, matched_game))
            to_train_pids.append(pid)
        else:
            stat_projections[pid] = prev_proj

    logger.info("Need to model: %s" % to_train_pids)

    # in parallel, train and project players that need it
    poolsize = 4
    args = iter(process_inputs)
    pool = mp.Pool(poolsize)
    pool_results = pool.imap_unordered(trainAndProjectPlayer, args)
    pool.close()
    pool.join()
    for pid,pid_results in pool_results:
        if pid_results is None:
            invalid_players.append(pid)
        else:
            stat_projections[pid] = pid_results

    logger.info("%s players projected, %s invalid" % (len(stat_projections), len(invalid_players)))
    point_projections, point_variances, point_trends = getFantasyPoints(stat_projections, kimono_info['point_values'])

    kimono_info['stat_projections'] = stat_projections
    kimono_info['point_projections'] = point_projections
    kimono_info['point_variances'] = point_variances
    kimono_info['point_trends'] = point_trends
    kimono_info['invalids'] = invalid_players

    # remove invalid players from roster construction
    for pid in invalid_players:
        kimono_info['player_salaries'].pop(pid, None)
        kimono_info['point_projections'].pop(pid, None)
        kimono_info['point_variances'].pop(pid, None)
        kimono_info['point_trends'].pop(pid, None)
        kimono_info['player_positions'].pop(pid, None)
        kimono_info['player_teams'].pop(pid, None)

    # create optimal matchup
    optros = FanDuelOptimalRoster(kimono_info['budget'], 
                           kimono_info['player_salaries'], 
                           kimono_info['point_projections'],
                           kimono_info['point_variances'],
                           kimono_info['player_positions'],
                           kimono_info['player_teams'],
                           kimono_info['roster_positions'])

    optimal = optros.constructOptimalByPoints()
    for i,opts in enumerate(optimal):
        # just get players
        only_player_rosters = []
        for opt in opts:
            only_players = [_[0] for _ in opt]
            only_player_rosters.append(only_players)
        kimono_info['roster_%s' % (i+1)] = only_player_rosters

    # save matchup info
    data = prepUpdate(game_id, kimono_info)
    data['update_time'] = datetime.now()
    data['_id'] = game_id
    data['targeturl'] = new_url
    nba_conn.saveDocument(upcoming_collection, data)

def getPreviousProjection(pid, matched_game):
    row = projection_collection.find_one({"player_id": pid, "time": matched_game['time']})
    if row:
        # average out the values
        vals = defaultdict(list)
        for pjs, stats in row['projections'].iteritems():
            for k,v in stats.iteritems():
                vals[k].append(v)
        avgs = {}
        for k,v in vals.items():
            mean_mean = np.mean([_[0] for _ in v])
            var_mean = ((1.0/len(v))**2) * sum([_[1] for _ in v])
            mean_mod = np.mean([_[2] for _ in v])
            avgs[k] = (mean_mean, var_mean, mean_mod)
        return avgs
    return None

def graphUpcomingGames(game_id):
    upcoming = upcoming_collection.find_one({"_id": game_id})
    players = list(set(upcoming['players']) - set(upcoming['invalids']))
    positions = upcoming['player_positions']
    salaries = upcoming['player_salaries']
    stat_projections = upcoming['stat_projections']
    pt_projections = upcoming['point_projections']
    pt_variances = upcoming['point_variances']
    pt_trends = upcoming['point_trends']
    ts = upcoming['update_time']
    avg_pts = lastTenGameFantasyAvg(players, ts, upcoming['point_values'])

    min_splits = teamPositionMinuteSplits(players, ts, positions)

    proj_by_position = defaultdict(dict)
    for k,proj in stat_projections.iteritems():
        pos = positions[k]
        proj_by_position[pos][k] = proj

    for pos, projs in proj_by_position.iteritems():
        # sort keys by projected points
        sorted_keys = sorted(projs.keys(), key=lambda x: pt_projections[x])
        labels = []
        mean_graph = []
        var_graph = []
        trend_graph = []
        usg_graph = []
        usg_var = []
        mp_graph = []
        mp_var = []
        avg_graph = []
        salaries_graph = []
        for k in sorted_keys:
            if pt_projections[k] < 10:
                continue
            labels.append("%s (%s)" % (k, salaries[k]))
            mean_graph.append(pt_projections[k])
            var_graph.append(math.sqrt(pt_variances[k]) * 2)
            trend_graph.append(pt_trends[k])
            usg_graph.append(projs[k]['USG%'][0])
            usg_var.append(projs[k]['USG%'][1])
            mp_graph.append(projs[k]['MP'][0])
            mp_var.append(projs[k]['MP'][1])
            avg_graph.append(avg_pts[k])
            salaries_graph.append(float(salaries[k]))
        # make regular plot
        fig = plt.figure()
        x = range(len(mean_graph))
        p2 = plt.errorbar(x, mean_graph, yerr=var_graph, label='proj')
        #p2 = plt.plot(mean_graph, label='proj')
        p_labels = plt.xticks(x, labels, rotation='vertical')
        p4 = plt.plot(trend_graph, marker='o', label='trend')
        p5 = plt.plot(usg_graph, marker='o', label='usg')
        p6 = plt.plot(mp_graph, marker='o', label='mp')
        p8 = plt.plot(avg_graph, marker='o', label='10DayAvg')
        p7 = plt.plot(x, [0.0]*len(x), label='zero')
        fig.suptitle(pos, fontsize=20)
        plt.legend(loc='best', shadow=True)

        # make per salary plot
        fig2 = plt.figure()
        x = range(len(mean_graph))
        p2 = plt.errorbar(x, mean_graph, yerr=var_graph, label='proj')
        p3 = plt.plot(trend_graph, marker='o', label='trend')
        p_labels = plt.xticks(x, labels, rotation='vertical')
        # calculate per $1000 stats
        mean_per = np.divide(mean_graph, salaries_graph) * 100.0
        usg_per = np.divide(usg_graph, salaries_graph) * 100.0
        mp_per = np.divide(mp_graph, salaries_graph) * 100.0
        avg_per = np.divide(avg_graph, salaries_graph) * 100.0
        p4 = plt.plot(mean_per, label='proj/$')
        p5 = plt.plot(usg_per, marker='o', label='usg/$')
        p6 = plt.plot(mp_per, marker='o', label='mp/$')
        p7 = plt.plot(x, [0.0]*len(x), label='zero')
        p8 = plt.plot(avg_per, marker='o', label='10DayAvg/$')
        fig2.suptitle("%s per $100" % pos, fontsize=20)
        plt.legend(loc='best', shadow=True)

    plt.show(block=True)
    
def lastTenGameFantasyAvg(pids, ts, point_vals):
    # calc stats
    stats = {}
    for pid in pids:
        pstats = defaultdict(list)
        results = player_game_collection.find({"player_id": pid,
                                               "game_time": {"$lt": ts}}, 
                                               sort=[("game_time",-1)], 
                                               limit=10)
        results = list(results)
        for row in results:
            for k,v in row.iteritems():
                pstats[k].append(v)
        avgs = {}
        for k in point_vals.keys():
            values = [_ for _ in pstats[k] if _ is not None]
            if len(values) == 0:
                avgs[k] = 0.0 # assuming that no point values for percentages
            else:
                avgs[k] = np.mean(values)
        stats[pid] = avgs
    # calc points
    fantasypts = {}
    for pid, pstat in stats.iteritems():
        pts = 0.0
        for k, w in point_vals.iteritems():
            stat_mean = pstat[k]
            pts += w*float(stat_mean)
        fantasypts[pid] = pts
    return fantasypts

def teamPositionMinuteSplits(pids, ts, positions):
    # find all the teams:
    team_ids = set()
    for pid in pids:
        # get team
        team_row = team_collection.find_one({"players": pid})
        if not team_row:
            raise Exception("Could not find team for %s" % pid)
        team_id = team_row['_id']
        team_ids.add(team_id)
    all_positions = set(positions.values())
    results = defaultdict(dict)
    for team_id in team_ids:
        for pos in all_positions:
            # find the last 5 games for the team
            games = game_collection.find({"teams": team_id, "time" : {"$lt" : ts}}, sort=[("time",-1)], limit=5)
            games = list(games)
            game_labels = [g["_id"] for g in games]
            if len(games) == 0:
                raise Exception("Could not find prior games for %s" % team_id)

            # find players who played on the same team and position
            validgames = []
            for game in games:
                gid = game['_id']
                playergames = player_game_collection.find({"game_id": gid, "player_team": team_id})
                curr_mins = {}
                for pgame in playergames:
                    player_id = pgame['player_id']
                    minutes = pgame['MP']
                    player_row = player_collection.find_one({"_id": player_id})
                    if pos in player_row['position']:
                        curr_mins[player_id] = minutes
                validgames.append(curr_mins)
            results[team_id][pos] = (game_labels, validgames)

    results = dict(results)
    for team_id, v in results.iteritems():
        for pos, (labels, games) in v.iteritems():
            all_players = [k for g in games for k in g.keys()]
            by_player = {p:[] for p in all_players}
            for v in games:
                for p in by_player.keys():
                    by_player[p].append(v.get(p,0.0))
            print by_player

            # make per salary plot
            fig = plt.figure()
            fig.suptitle("%s: %s" % (team_id, pos), fontsize=20)
            x = range(len(labels))
            p_labels = plt.xticks(x, labels, rotation='vertical')
            for p, values in by_player.iteritems():
                plt.plot(values, marker='o', label=p)
            plt.legend(loc='best', shadow=True)
    plt.show(block=True)


def compareHistoricalGame(game_id, point_values=None, graph=False):
    '''
    Fails for players who have since left either team
    '''
    if point_values is None:
        point_values = FANDUEL_PT_VALUES

    game_row = game_collection.find_one({"_id": game_id})
    if not game_row:
        raise Exception("Could not find game %s" % game_id)    
    playerstats = player_game_collection.find({"game_id": game_id})

    # mock matched game
    home_id = game_row['team2_id']
    away_id = game_row['team1_id']
    home_row = team_collection.find_one({"_id": home_id})
    away_row = team_collection.find_one({"_id": away_id})
    if not home_row:
        raise Exception("Could not find %s" % home_id)
    if not away_row:
        raise Exception("Could not find %s" % away_id)
    home_name = home_row['name']
    away_name = away_row['name']
    matched_game = {"_id" : "%s@%s" % (away_id,home_id),
                    "away_id" : away_id,
                    "home_team_name" : home_name,
                    "teams" : [away_id, home_id],
                    "home_id" : home_id,
                    "away_team_name" : away_name,
                    "time" : game_row['time']
                    }

    data = {}
    stat_projections = {}
    for pstat in playerstats:
        pid = pstat['player_id']
        prev_proj = getPreviousProjection(pid, matched_game)
        if not prev_proj:
            print "No projections found for %s" % pid
            pid, prev_proj = trainAndProjectPlayer((pid, matched_game))
        if prev_proj:
            stat_projections[pid] = prev_proj
            # calculate player fantasy points
            actual_pts = sum([v*float(pstat.get(k,0.0)) for k,v in point_values.iteritems()])
            data[pid] = {'actual':actual_pts}
        else:
            print "Failed to project for %s" % pid
            continue

    # extract usg% and mp
    usg = {}
    mp = {}
    for pid, proj in stat_projections.iteritems():
        usg[pid] = stat_projections[pid]['USG%']
        mp[pid] = stat_projections[pid]['MP']

    # calculate points
    point_projections, point_variances, point_trends = getFantasyPoints(stat_projections, point_values)
    for pid in point_projections.keys():
        data[pid]['proj_mean'] = point_projections[pid]
        data[pid]['proj_var'] = math.sqrt(point_variances[pid]) * 2
        data[pid]['proj_trend'] = point_trends[pid]
        data[pid]['mp'] = usg[pid][0]
        data[pid]['mp_var'] = math.sqrt(usg[pid][1]) * 2
        data[pid]['usg'] = mp[pid][0]
        data[pid]['usg_var'] = math.sqrt(mp[pid][1]) * 2

    if graph:
        # sort keys by projected points
        sorted_keys = sorted(data.keys(), key=lambda x: data[x]['proj_mean'])
        mean_graph = []
        var_graph = []
        real_graph = []
        trend_graph = []
        usg_graph = []
        usg_var = []
        mp_graph = []
        mp_var = []
        for k in sorted_keys:
            mean_graph.append(data[k]['proj_mean'])
            var_graph.append(data[k]['proj_var'])
            real_graph.append(data[k]['actual'])
            trend_graph.append(data[k]['proj_trend'])
            usg_graph.append(data[k]['usg'])
            usg_var.append(data[k]['usg_var'])
            mp_graph.append(data[k]['mp'])
            mp_var.append(data[k]['mp_var'])
        fig = plt.figure()
        x = range(len(mean_graph))
        p2 = plt.errorbar(x, mean_graph, yerr=var_graph, label='proj')
        #p2 = plt.plot(mean_graph, label='proj')
        p_labels = plt.xticks(x, sorted_keys, rotation='vertical')
        p3 = plt.plot(real_graph, 'k:', marker='o', label='real')
        p4 = plt.plot(trend_graph, marker='o', label='trend')
        p5 = plt.plot(usg_graph, marker='o', label='usg')
        p6 = plt.plot(mp_graph, marker='o', label='mp')
        p7 = plt.plot(x, [0.0]*len(x), label='zero')
        fig.suptitle(game_id, fontsize=20)
        plt.legend(loc='best', shadow=True)
        plt.show(block=True)
    return data

def trainAndProjectPlayer(args):
    """
    args must be a tuple of (pid, matched_game, output_queue)
    """
    pid, matched_game = args
    game_time = matched_game['time']

    # get upcoming game features for player
    try:
        upcoming_features = getUpcomingGameFeatures(pid, matched_game)
    except Exception as e:
        print "Error getting upcoming features for %s: %s" % (pid, e)
        print traceback.print_exc()
        return (pid,None)

    # get player models
    try:
        player_models = trainModelsForPlayer(pid)
    except Exception as e:
        print "Error training models for %s: %s" % (pid, e)
        print traceback.print_exc()
        return (pid,None)

    # make projections
    pproj = {}
    for y_key, (core_processors, core_models, trend_processors, trend_models) in player_models.iteritems():
        mean, variance, trend, trend_variance = makeProjection(upcoming_features, core_processors, core_models, trend_processors, trend_models)
        pproj[y_key] = (mean, variance, trend, trend_variance)

    print "%s: %s" % (pid, pproj)

    # save projection
    try:
        ts_format = '%Y-%m-%d %H:%M:%S'
        current = datetime.fromtimestamp(time.time()).strftime(ts_format)
        row = projection_collection.find_one({"player_id": pid, "time": game_time})
        if row:
            old_proj = copy.deepcopy(row['projections'])
            old_proj[current] = pproj
            # choose most recent 5 runs
            all_keys = [datetime.strptime(k, ts_format) for k in old_proj.keys()]
            most_recent = list(sorted(all_keys, reverse=True))[:5]
            most_recent_keys = [_.strftime(ts_format) for _ in most_recent]
            new_proj = {k: old_proj[k] for k in most_recent_keys}
            row['projections'] = new_proj
        else:
            row = {"player_id": pid,
                   "time": game_time,
                   "projections": {current: pproj}}
        print 'Saving %s' % row
        nba_conn.saveDocument(projection_collection, row)
    except Exception as e:
        print 'Exception saving projection: %s' % e
        print traceback.print_exc()

    return (pid, pproj)


def makeProjection(new_features, core_processors, core_models, trend_processors, trend_models):
    '''
    Using:

    C := core projection
    M := vector of trends
    Z := final projection
    W := 1/Var(M)
    Z = C(W*M)
    Var(Z) = Var(C)[(W^2)Var(M)]


    TODO: cut off gaussian projections at 0 and recalculate mean/var ??
    '''
    core_projections = {}
    trend_projections = {}
    try:
        # make core projections
        for name, model in core_models.iteritems():
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = new_features[name]
            proc = core_processors[name]
            #print "transforming: %s" % (zip(cont_labels,cont_features),)
            new_sample = np.array([proc.transform(cont_features, cat_features)])
            mean, variance = model.predict(new_sample)
            mean = max(0.0, mean[0][0])
            variance = abs(variance[0][0])
            core_projections[name] = (mean, variance)
        # make trend projections
        for name, model in trend_models.iteritems():
            cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = new_features[name]
            proc = trend_processors[name]
            new_sample = np.array([proc.transform(cont_features, cat_features)])
            mean, variance = model.predict(new_sample)
            mean = max(0.0,mean[0][0])
            variance = abs(variance[0][0])
            trend_projections[name] = (mean, variance)
    except Exception as e:
        print "Error making projections: %s" % e
        print traceback.print_exc()
        raise e

    # modulate core by each trend projection
    # DEFAULTING TO USING PLAYER FEATURES AS CORE
    core_mean, core_var = core_projections['plf']

    # mix projections together, weighted by variance
    mods = trend_projections.values()
    mod_means = np.array([_[0] for _ in mods])
    mod_vars = np.array([_[1] for _ in mods]) 
    # deal with zeros
    zero_replace = 0.00001
    non_zero_vars = mod_vars[mod_vars > 0.0]
    if len(non_zero_vars) > 0:
        zero_replace = min(zero_replace, min(non_zero_vars))
    mod_vars[mod_vars == 0.0] = zero_replace

    #print "core_mean: %s, core_var: %s" % (core_mean,core_var)
    #print "modmeans: %s, modvars: %s" % (mod_means, mod_vars)

    mod_weights = np.array([1.0/_ for _ in mod_vars])
    mod_weights = mod_weights / sum(mod_weights)
    weighted_mods_avg = np.average(mod_means, weights=mod_weights)
    avg = weighted_mods_avg * core_mean
    print "core: %s, mod: %s, avg: %s" % (core_mean,weighted_mods_avg,avg)

    # calculate Var(Z)
    w_2 = [_**2 for _ in mod_weights]
    mod_variance = np.inner(w_2,mod_vars)
    mod_variance_avg = ((1.0/len(mod_vars))**2) * mod_variance
    # Var(XY) = (E(X)^2)*(Var(Y)) + (E(Y)^2)*(Var(X)) + Var(X)*Var(Y)
    total_variance = (core_mean**2)*(mod_variance_avg) + (weighted_mods_avg**2)*(core_var) + (mod_variance_avg*core_var)

    return (avg, total_variance, weighted_mods_avg, mod_variance_avg)

def prepUpdate(game_id, kimono_info):
    '''
    Pop unnecessary items from kimono_info
    Check if optimal roster is different, if so, log
    '''
    kimono_info.pop('player_teams', None)
    kimono_info.pop('player_games', None)
    kimono_info.pop("player_dicts", None)
    kimono_info['update_needed'] = False
    saved = upcoming_collection.find_one({"_id": game_id})
    if saved:
        saved.update(kimono_info)
        return saved
    else:
        kimono_info['update_needed'] = True
        print "%s: new players: %s" % (game_id, kimono_info['players'])
        print "%s: new optimal roster: %s" % (game_id, kimono_info['roster_1'])
        print "%s: new invalids: %s" % (game_id, kimono_info['invalids'])
        print "%s: new projections: %s" % (game_id, kimono_info['point_projections'])
        return kimono_info

def mergeFanduelDB(game_id, kimono_info):
    '''
    Merge db row if exists, and augment with static info
    '''
    saved = upcoming_collection.find_one({"_id": game_id})
    if saved:
        kimono_info['manual_invalids'] = saved.get('manual_invalids',[])
        kimono_info['manual_valids'] = saved.get('manual_valids', [])
    else:
        kimono_info['manual_invalids'] = []
        kimono_info['manual_valids'] = []
    kimono_info['point_values'] = FANDUEL_PT_VALUES
    return kimono_info


def matchUpcomingGame(home_abbr, away_abbr):
    # * to BR translations
    home_abbr = TO_BR_TEAM_TRANSLATIONS.get(home_abbr, home_abbr)
    away_abbr = TO_BR_TEAM_TRANSLATIONS.get(away_abbr, away_abbr)
    # TODO: change timedelta back to one day
    upcoming_games = future_collection.find({"time": {"$gt" : datetime.now() - timedelta(2)}}, sort=[("time",1)])
    for game_dict in upcoming_games:
        away_id = game_dict["away_id"]
        home_id = game_dict["home_id"]
        # sometimes the fanduel team abbr are shorter than 3 chars, ie. SA
        if away_abbr == away_id and home_abbr == home_id:
            return game_dict
    raise Exception("Could not match game %s @ %s" % (away_abbr, home_abbr))

def availablePlayers(upcoming_game_dict):
    away_name = upcoming_game_dict['away_team_name']
    home_name = upcoming_game_dict['home_team_name']
    away_info = teamInfo(away_name)
    home_info = teamInfo(home_name)
    away_players = away_info['players']
    home_players = home_info['players']
    players = {away_name : away_players,
               home_name : home_players}
    return players

def teamInfo(team_name):
    result = team_collection.find_one({"name":str(team_name).strip()})
    return result


def trainModelsForPlayer(pid):
    to_return = {}
    training_games = findAllTrainingGames(pid, limit=40)
    for y_key in Y_KEYS:
        core_processors, core_models = trainCoreModels(pid, training_games, y_key, weigh_recent=True, test=False, plot=False)
        trend_processors, trend_models = trainTrendModels(pid, training_games, y_key, weigh_recent=True, test=False, plot=False)
        to_return[y_key] = (core_processors, core_models, trend_processors, trend_models)
    return to_return


def getFantasyPoints(projections, point_vals):
    points = {}
    variances = {}
    trends = {}
    for pid, proj in projections.iteritems():
        pts_var = 0.0
        pts = 0.0
        unmodded = 0.0
        for k, w in point_vals.iteritems():
            stat_mean = proj[k][0]
            stat_var = proj[k][1]
            stat_trend = proj[k][2]
            pts += w*float(stat_mean)
            unmodded += w*float(stat_mean/stat_trend)
            pts_var += w**2 * float(stat_var)
        variances[pid] = pts_var
        points[pid] = pts
        trends[pid] = pts - unmodded
    return points, variances, trends

def addPlayerToInvalid(game_id, player_id):
    # get row
    row = upcoming_collection.find_one({"_id": game_id})
    if not row:
        row = {"_id" : game_id}
    invalids = row.get('manual_invalids',[])
    invalids.append(player_id)
    invalids = list(set(invalids))
    row['manual_invalids'] = invalids
    nba_conn.saveDocument(upcoming_collection, row)

if __name__ == "__main__":
    #comps = compareHistoricalGame("201502030POR", graph=True)
    graphUpcomingGames("11732?tableId=10901317")
