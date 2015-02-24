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
import numpy as np
import traceback

from statsETL.bball.NBAcrawler import upcomingGameCrawler
from statsETL.db.mongolib import *
from analysis.bball.train import *
from analysis.bball.playerAnalysis import featureExtractor, findAllTrainingGames
from analysis.bball.rosterAnalysis import FanDuelOptimalRoster
from analysis.util.kimono import *


def analyzeFanDuelGame(game_id, window_size=15):
    """
    create new game row in db
    create kimono api, continuously update scraped values (LOG)
    train models or get already trained ones (LOG to ?dynamo?)
    continuously update optimal roster + save to db (LOG UPDATES)
    """
    new_url = createFanDuelDraftURL(game_id)

    print new_url

    kimono_info = fanDuelNBADraftAPIContent(new_url)
    kimono_info = mergeFanduelDB(game_id, kimono_info)
    
    # build players by game lookup
    players_by_game = defaultdict(list)
    for pid, game in kimono_info['player_games'].iteritems():
        players_by_game[game].append(pid)

    # sync upcoming games
    crawlUpcomingGames(days_ahead=7)
    valid_game_ids = set(kimono_info['player_games'].values())

    # kimono invalid players, adjusted by manual in/validation
    invalid_players = list(set(kimono_info['invalids'] + kimono_info['manual_invalids']))
    for pid in kimono_info['manual_valids']:
        if pid in invalid_players:
            invalid_players.pop(pid)
    print "invalids: %s" % invalid_players
    print "GTD: %s" % kimono_info['gtds']

    # match up game ids with upcoming games
    upcoming_by_id = {} 
    player_teams = {}
    for gid in valid_game_ids:
        teams = gid.split('@')
        away = teams[0]
        home = teams[1]
        matched_game = matchUpcomingGame(home, away)

        print "Matched %s with %s" % (gid, matched_game)

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
                print "%s not on %s or %s rosters" % (p,home_name,away_name)
                invalid_players.append(p)
    kimono_info['player_teams'] = player_teams

    # train model for players and make predictions
    stat_projections = {}
    for pid in kimono_info['players']:
        # skip invalid players
        if pid in invalid_players:
            print "%s is invalid, skipping" % pid
            continue

        # get the matched game
        gid = kimono_info['player_games'][pid]
        matched_game = upcoming_by_id[gid]

        # get upcoming game features for player
        upcoming_features = getUpcomingGameFeatures(pid, matched_game)

        # train player models
        try:
            player_models = trainModelsForPlayer(pid)
        except Exception as e:
            print "Error training models for %s: %s" % (pid, e)
            traceback.print_exc()
            invalid_players.append(pid)
            continue

        # make projections
        pproj = {}
        for y_key, (core_processors, core_models, trend_processors, trend_models) in player_models.iteritems():
            mean, variance = makeProjection(upcoming_features, core_processors, core_models, trend_processors, trend_models)
            pproj[y_key] = (mean, variance)
        print pproj
        stat_projections[pid] = pproj

    point_projections, point_variances = getFantasyPoints(stat_projections, kimono_info['point_values'])

    kimono_info['stat_projections'] = stat_projections
    kimono_info['point_projections'] = point_projections
    kimono_info['point_variances'] = point_variances
    kimono_info['invalids'] = invalid_players

    # remove invalid players
    for pid in invalid_players:
        kimono_info['player_salaries'].pop(pid, None)
        kimono_info['point_projections'].pop(pid, None)
        kimono_info['point_variances'].pop(pid, None)
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

    optimal = optros.constructOptimal()
    for i,opt in enumerate(optimal):
        # just get players
        only_players = [_[0] for _ in opt]
        kimono_info['roster_%s' % (i+1)] = only_players

    # save matchup info
    data = prepUpdate(game_id, kimono_info)
    data['update_time'] = datetime.now()
    data['_id'] = game_id
    data['targeturl'] = new_url
    nba_conn.saveDocument(upcoming_collection, data)


def makeProjection(new_features, core_processors, core_models, trend_processors, trend_models):
    '''
    Using:

    C := core projection
    M := vector of trends
    Z := final projection
    W := 1/Var(M)
    Z = C(W*M)
    Var(Z) = Var(C)[(W^2)Var(M)]
    '''
    core_projections = {}
    trend_projections = {}
    # make core projections
    for name, model in core_models.iteritems():
        cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = new_features[name]
        proc = core_processors[name]
        new_sample = proc.transform(cont_features, cat_features)
        mean, variance = model.predict(np.array([new_sample]))
        mean = max(0.0, mean[0][0])
        variance = max(0.0, variance[0][0])
        core_projections[name] = (mean, variance)
    # make trend projections
    for name, model in trend_models.iteritems():
        cat_labels, cat_features, cont_labels, cont_features, cat_feat_splits = new_features[name]
        proc = trend_processors[name]
        new_sample = proc.transform(cont_features, cat_features)
        mean, variance = model.predict(np.array([new_sample]))
        mean = max(0.0,mean[0][0])
        variance = max(0.0, variance[0][0])
        trend_projections[name] = (mean, variance)
    # modulate core by each trend projection
    # DEFAULTING TO USING PLAYER FEATURES AS CORE
    core_mean, core_var = core_projections['plf']
    projections = {}
    for name, (trend_mean, trend_var) in trend_projections.iteritems():
        # Var(XY) = (E(X)^2)*(Var(Y)) + (E(Y)^2)*(Var(X)) + Var(X)*Var(Y)
        new_var = (trend_mean**2)*(core_var) + (core_mean**2)*(trend_var) + (trend_var*core_var)
        projections[name] = (trend_mean*core_mean,new_var)
    
    # mix projections together, weighted by variance
    mods = trend_projections.values()
    mod_means = np.array([_[0] for _ in mods])
    mod_vars = np.array([_[1] for _ in mods])


    mod_weights = np.array([1.0/_ for _ in mod_vars])
    mod_weights = mod_weights / sum(mod_weights)
    weighted_mods_avg = np.average(mod_means, weights=mod_weights)
    print "core: %s, mod: %s" % (core_mean,weighted_mods_avg)
    avg = weighted_mods_avg * core_mean

    # calculate Var(Z)
    w_2 = [_**2 for _ in mod_weights]
    mod_variance = np.inner(w_2,mod_vars)
    mod_variance_avg = ((1.0/len(mod_vars))**2) * mod_variance
    total_variance = (core_mean**2)*(mod_variance_avg) + (weighted_mods_avg**2)*(core_var) + (mod_variance_avg*core_var)

    return (avg,total_variance)

def crawlUpcomingGames(days_ahead=7):
    dates = []
    new_games = []
    for i in range(days_ahead):
        d = datetime.now() + timedelta(i)
        gl_crawl = upcomingGameCrawler(date=d)
        new_games = gl_crawl.crawlPage()
    return new_games

def prepUpdate(game_id, kimono_info):
    '''
    Pop unnecessary items from kimono_info
    Check if optimal roster is different, if so, log
    '''
    kimono_info.pop('player_teams', None)
    kimono_info.pop('player_positions', None)
    kimono_info.pop('player_salaries', None)
    kimono_info.pop('stat_projections', None)
    kimono_info.pop('player_games', None)
    kimono_info.pop("player_dicts", None)
    kimono_info['update_needed'] = False
    saved = upcoming_collection.find_one({"_id": game_id})
    if saved:
        if saved.get('players') != kimono_info['players']:
            print "%s: updated players: %s" % (game_id, kimono_info['players'])
        if saved.get('roster_1') != kimono_info['roster_1']:
            print "%s: updated optimal roster: %s" % (game_id, kimono_info['roster_1'])
            kimono_info['update_needed'] = True
        if saved.get('invalids') != kimono_info['invalids']:
            print "%s: updated invalids: %s" % (game_id, kimono_info['invalids'])
        if saved.get('point_projections') != kimono_info['point_projections']:
            print "%s: updated projections: %s" % (game_id, kimono_info['point_projections'])
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
    kimono_info['point_values'] = {'TOV': -1.0, 
                                   'AST': 1.5,
                                   'STL': 2.0,
                                   'TRB': 1.2,
                                   'BLK': 2.0,
                                   'PTS': 1.0}
    return kimono_info


def matchUpcomingGame(home_abbr, away_abbr):
    # fanduel to nba.com translations
    translations = {'SA': 'SAS',
                    'GS': 'GSW',
                    'NO': 'NOP'
                    }
    home_abbr = translations.get(home_abbr, home_abbr)
    away_abbr = translations.get(away_abbr, away_abbr)
    upcoming_games = future_collection.find({"time": {"$gt" : datetime.now()}}, sort=[("time",1)])
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
    training_games = findAllTrainingGames(pid)
    to_return = {}
    for y_key in ['PTS','TRB','AST','TOV','STL','BLK','MP']:
        core_processors, core_models = trainCoreModels(pid, training_games, y_key, weigh_recent=True, test=False, plot=False)
        trend_processors, trend_models = trainTrendModels(pid, training_games, y_key, weigh_recent=True, test=False, plot=False)
        to_return[y_key] = (core_processors, core_models, trend_processors, trend_models)
    print to_return.keys()
    return to_return


def getFantasyPoints(projections, point_vals):
    points = {}
    variances = {}
    for pid, proj in projections.iteritems():
        pts_var = 0.0
        pts = 0.0
        for k,(m,v) in proj.iteritems():
            w = point_vals[k]
            pts += w**float(m)
            pts_var += w**2 * float(v)
        variances[pid] = pts_var
    return points, variances


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
    game_id = '11680?tableId=10727641'
    analyzeFanDuelGame(game_id)