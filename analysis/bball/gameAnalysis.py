'''
maintaining db and running jobs for upcoming games that are bet on
maintaining a db with roster, salary, payroll, scoring for all active games
(TODO: create kimono interface)
matching rosters up with db players
training models + making predictions for all stats X players
converting predictions to projected points
make best roster with salary, payroll, and projected points
(TODO: maintain scoring functions for df and fd)
(TODO: test roster algorithm)

get automatic refresher on game sites

'''
import copy
import itertools
from collections import defaultdict
from datetime import datetime

from statsETL.bball.NBAcrawler import upcomingGameCrawler
from statsETL.db.mongolib import *
from analysis.bball.train import DataPreprocessor, RandomForestValidationFramework
from analysis.bball.playerAnalysis import featureExtractor, findAllTrainingGames
from analysis.bball.rosterAnalysis import FanDuelOptimalRoster
from analysis.util.kimono import *



def analyzeFanDuelGame(game_id):
    """
    create new game row in db
    create kimono api, continuously update scraped values (LOG)
    train models or get already trained ones (LOG to ?dynamo?)
    continuously update optimal roster + save to db (LOG UPDATES)
    """
    # rewrite the target url to kimono fanduel nba api (if necessary), then recrawl
    kimono_info = fanDuelNBADraftAPIContent()
    kimono_info = mergeFanduelDB(game_id, kimono_info)
    
    # build players by game lookup
    players_by_game = defaultdict(list)
    for pid, game in kimono_info['player_games'].iteritems():
        players_by_game[game].append(pid)

    # match up with upcoming games
    # TODO: keep upcoming games DB
    upcoming_games = upcomingGames(date=kimono_info['game_dates'])
    valid_game_ids = set(kimono_info['player_games'].values())

    # invalid players
    invalid_players = []

    # match up game ids with upcoming games
    upcoming_by_id = {} 
    player_teams = {}
    for game_id in valid_game_ids:
        teams = game_id.split('@')
        away = teams[0]
        home = teams[1]
        matched_game = matchUpcomingGame(upcoming_games, home, away)
        upcoming_by_id[game_id] = matched_game

        # find players in game
        game_pids = players_by_game[game_id]
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

        # get matched up
        game_id = kimono_info['player_games'][pid]
        matched_game = upcoming_by_id[game_id]

        try:
            # get upcoming game features for player
            pfe = featureExtractor(pid, target_game=matched_game)
            cat_labels, cat_features, cont_labels, cont_features = pfe.run()
            # train player models
            pprocessors, pmodels = trainModelsForPlayer(pid)
        except Exception as e:
            print "Error training models for %s: %s" % (pid, e)
            invalid_players.append(pid)
            continue
        # make projections
        pproj = {}
        for k,v in pmodels.iteritems():
            proc = pprocessors[k]
            model = pmodels[k]
            new_sample = proc.transform(cont_features, cat_features)
            pproj[k] = model.predict(new_sample)
        stat_projections[pid] = pproj
    point_projections = getFantasyPoints(stat_projections, kimono_info['point_values'])

    print "invalid: %s" % invalid_players
    print point_projections

    kimono_info['stat_projections'] = stat_projections
    kimono_info['point_projections'] = point_projections

    # remove invalid players
    for pid in invalid_players:
        kimono_info['player_salaries'].pop(pid, None)
        kimono_info['point_projections'].pop(pid, None)
        kimono_info['player_positions'].pop(pid, None)
        kimono_info['player_teams'].pop(pid, None)

    print kimono_info

    # create optimal matchup
    optros = FanDuelOptimalRoster(kimono_info['budget'], 
                           kimono_info['player_salaries'], 
                           kimono_info['point_projections'], 
                           kimono_info['player_positions'],
                           kimono_info['player_teams'],
                           kimono_info['roster_positions'])
    optimal = optros.constructOptimal()
    kimono_info['optimal_roster'] = optimal

    # save matchup info
    # TODO: CHECK IF ROW EXISTS, AND WHICH INFORMATION IS NEW
    kimono_info['update_time'] = datetime.now()
    nba_conn.saveDocument(upcoming_collection, kimono_info)


def mergeFanduelDB(game_id, kimono_info):
    '''
    Merge db row if exists, and augment with static info
    '''
    kimono_info['point_values'] = {'TOV': -1.0, 
                                   'AST': 1.5,
                                   'STL': 2.0,
                                   'TRB': 1.2,
                                   'BLK': 2.0,
                                   'PTS': 1.0}
    return kimono_info


def upcomingGames(date=None):
    if date:
        if not isinstance(date,list):
            date = [date]
    else:
        date = [datetime.now() + timedelta(int(1))]
    games = []
    for d in date:
        gl_crawl = upcomingGameCrawler(date=d)
        new_games = gl_crawl.crawlPage()
        games = games + new_games
    return games

def matchUpcomingGame(upcoming_games, home_abbr, away_abbr):
    for game_dict in upcoming_games:
        away_name = game_dict['away_team_name']
        home_name = game_dict['home_team_name']
        away_info = teamInfo(away_name)
        home_info = teamInfo(home_name)
        away_id = away_info["_id"]
        home_id = home_info["_id"]
        if away_id == away_abbr or home_id == home_abbr:
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
    print pid
    trained_models = {}
    processors = {}
    playergames = findAllTrainingGames(pid)
    
    failed_games = []
    cat_X = []
    cont_X = []
    cat_labels = None
    cont_labels = None

    # if no training stats, don't train model
    if not playergames:
        raise Exception("No training data")

    for g in playergames:
        gid = str(g['game_id'])
        try:
            fext = featureExtractor(pid, target_game=gid)
            cat_labels, cat_features, cont_labels, cont_features = fext.run()
            cat_X.append(cat_features)
            cont_X.append(cont_features)
        except Exception as e:
            print "Error for %s: %s " % (gid,e)
            failed_games.append(gid)

    for y_key in ['PTS','TRB','AST','TOV','STL','BLK']:
        Y = []
        for g in playergames:
            gid = str(g['game_id'])
            if gid in failed_games:
                continue
            fext = featureExtractor(pid, target_game=gid)
            y = fext.getY(y_key)
            Y.append(y)

        # normalize the data
        proc = DataPreprocessor(cont_labels, 
                                copy.deepcopy(cont_X), 
                                cat_labels, 
                                copy.deepcopy(cat_X), 
                                Y)
        X = proc.getAllSamples()
        Y = proc.getAllY()
        labels = proc.getFeatureLabels()
        processors[y_key] = proc

        rf_framework = RandomForestValidationFramework(X,Y, test_size=0.10, feature_labels=labels)
        rf_framework.train()
        #rf_framework.test()
        trained_models[y_key] = rf_framework
    return (processors, trained_models)

def getFantasyPoints(projections, point_vals):
    points = {}
    for pid, proj in projections.iteritems():
        pts = 0.0
        for k,v in proj.iteritems():
            pts += float(v) * point_vals[k]
        points[pid] = pts
    return points


if __name__ == "__main__":
    data = analyzeFanDuelGame(None)
    print data
