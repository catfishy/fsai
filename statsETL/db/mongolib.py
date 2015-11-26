import sys
import simplejson as json
import pandas as pd

from pymongo import MongoClient
from bson.objectid import ObjectId

# TODO: move config params to different file

MONGOD_HOST = 'localhost'
MONGOD_PORT = 27017
NBA_DB = "nba_db"


class MongoConn:

    def __init__(self, db=None):
        if not db:
            raise Exception("Must specify DB to connect to")
        # create mongo client
        self.client = MongoClient(MONGOD_HOST, MONGOD_PORT)
        self.db = self.client[db]

    def getCollection(self, coll_name):
        return self.db[coll_name]

    def insertDocument(self, coll, doc):
        doc_id = coll.insert(doc)
        return doc_id

    def updateDocument(self, coll, row_id, doc, upsert=False):
        filt = {"_id": row_id}
        result = self.updateDocumentByFilter(coll, filt, doc, upsert=upsert)
        return result

    def updateDocumentByFilter(self, coll, filt, doc, upsert=False):
        result = coll.update_one(filt, {'$set' : doc}, upsert=upsert)
        return result

    def saveDocument(self, coll, doc):
        '''
        overwrites doc if found
        '''
        doc_id = coll.save(doc)
        return doc_id

    def removeDocument(self, coll, spec_or_id):
        if not spec_or_id:
            raise Exception("Must specify spec or ID")
        result = coll.remove(spec_or_id)
        return result

    def findOne(self, coll, query=None):
        if not query:
            result = coll.find_one()
        else:
            result = coll.find_one(query)
        return result

    def findByID(self, coll, doc_id):
        result = coll.find_one({"_id": doc_id})
        return result

    def find(self, coll, query, sort=None):
        '''
        returns a cursor object, can get number of results with
        results.count()
        '''
        if not query or type(query) != dict:
            raise Exception("Must specify a query in dict-form")
        results = coll.find(query)
        if sort:
            sorted_results = results.sort(sort)
            return sorted_results
        else:
            return results

    def createIndex(self, coll, index_args):
        '''
        example index args (for compound index):
        index_args = [("date", DESCENDING), ("author", ASCENDING)]
        '''
        index = coll.create_index(index_args)
        return index

    def ensureIndex(self, coll, index_args, unique=False, sparse=False):
        index = coll.create_index(index_args, unique=unique, sparse=sparse)
        return index

    def bulkInsert(self, coll, docs):
        if type(docs) != list:
            raise Exception("Documents arg must be a list of dicts")
        doc_ids = coll.insert(docs)
        return doc_ids

    def count(self, coll):
        return coll.count()

    def parallelScan(self, coll, num_cursors=1):
        cursors = coll.parallel_scan(num_cursors)
        return cursors




# get nba db conn
nba_conn = MongoConn(db=NBA_DB)

# old
upcoming_collection = nba_conn.getCollection("upcoming") # for upcoming bets
advanced_collection = nba_conn.getCollection("advanced")
onoff_collection = nba_conn.getCollection("onoff")
two_man_collection = nba_conn.getCollection("two_man")

# deprecate
'''
espn_stat_collection = nba_conn.getCollection("espnstats") # for espn team stats
espn_player_stat_collection = nba_conn.getCollection("espnplayerstats")
game_collection = nba_conn.getCollection("games")
player_collection = nba_conn.getCollection("players")
player_game_collection = nba_conn.getCollection("player_games")
team_game_collection = nba_conn.getCollection("team_games")

nba_conn.ensureIndex(team_collection, [("url", 1)])
nba_conn.ensureIndex(player_collection, [("url", 1)])
nba_conn.ensureIndex(player_collection, [("nba_id", 1)])
nba_conn.ensureIndex(team_game_collection, [("team_id", 1)])
nba_conn.ensureIndex(team_game_collection, [("game_id", 1)])
nba_conn.ensureIndex(player_game_collection, [("player_id", 1)])
nba_conn.ensureIndex(player_game_collection, [("game_id", 1)])
nba_conn.ensureIndex(game_collection, [('url',1)])
nba_conn.ensureIndex(team_game_collection, [("team_id", 1),("game_id", 1)], unique=True)
nba_conn.ensureIndex(player_game_collection, [("player_id", 1),("game_id", 1)], unique=True)
nba_conn.ensureIndex(espn_stat_collection, [('time', 1)], unique=True)
nba_conn.ensureIndex(espn_player_stat_collection, [("player_id", 1),('time', 1)], unique=True)
'''

# current
nba_teams_collection = nba_conn.getCollection("nba_teams")
shot_chart_collection = nba_conn.getCollection("shot_chart")
depth_collection = nba_conn.getCollection("depthchart")
shot_collection = nba_conn.getCollection("shot")
defense_collection = nba_conn.getCollection("defense")
rebound_collection = nba_conn.getCollection("rebound")
pass_collection = nba_conn.getCollection("pass")
nba_games_collection = nba_conn.getCollection("nba_games")
nba_players_collection = nba_conn.getCollection("nba_players")
# team stats vectors
nba_season_averages_collection = nba_conn.getCollection("nba_season_averages")
nba_team_vectors_collection = nba_conn.getCollection("nba_team_vectors")
# player stats vectors
nba_player_vectors_collection = nba_conn.getCollection("nba_player_vectors")
nba_against_vectors_collection = nba_conn.getCollection("nba_against_vectors")
nba_split_vectors_collection = nba_conn.getCollection("nba_split_vectors")
# util tables
nba_stat_ranges_collection = nba_conn.getCollection("nba_stat_ranges")

# indices
nba_conn.ensureIndex(nba_teams_collection, [("team_id", 1),("season", 1)], unique=True)
nba_conn.ensureIndex(depth_collection, [("team_id", 1),("season", 1)], unique=True)
nba_conn.ensureIndex(advanced_collection, [("player_id", 1),('time', 1),('team_id', 1)], unique=True)
nba_conn.ensureIndex(onoff_collection, [("player_id", 1),('time', 1),('team_id', 1)], unique=True)
nba_conn.ensureIndex(two_man_collection, [("player_one", 1),("player_two", 1),('time', 1),('team_id', 1)], unique=True, sparse=True)
nba_conn.ensureIndex(shot_chart_collection, [("player_id", 1),("game_id", 1),("time", 1)], unique=True)
nba_conn.ensureIndex(shot_chart_collection, [("game_id", 1)])
nba_conn.ensureIndex(shot_collection, [("player_id", 1),("game_id", 1),("time", 1)], unique=True)
nba_conn.ensureIndex(shot_collection, [("game_id", 1)])
nba_conn.ensureIndex(defense_collection, [("player_id", 1),("year", 1)], unique=True)
nba_conn.ensureIndex(rebound_collection, [("player_id", 1),("year", 1)], unique=True)
nba_conn.ensureIndex(pass_collection, [("player_id", 1),("year", 1)], unique=True)
nba_conn.ensureIndex(nba_games_collection, [("date", 1),("teams", 1)], unique=True)
nba_conn.ensureIndex(nba_games_collection, [("date", 1),("players", 1)])
nba_conn.ensureIndex(nba_season_averages_collection, [("date", 1), ("team_id", 1), ("window", 1)], unique=True)
nba_conn.ensureIndex(nba_player_vectors_collection, [("date", 1), ("player_id", 1), ("game_id",1), ("team_id",1), ("window", 1)], unique=True)
nba_conn.ensureIndex(nba_against_vectors_collection, [("game_id", 1), ("team_id", 1), ("window", 1)], unique=True)
nba_conn.ensureIndex(nba_split_vectors_collection, [("date", 1), ("player_id", 1)], unique=True)
nba_conn.ensureIndex(nba_stat_ranges_collection, [("date", 1),("vector_type", 1)], unique=True)


def getGameData(game_id):
    row = espn_games_collection.find_one({"_id": game_id})
    if not row:
        print "Could not find game %s" % game_id
        return
    data = {}
    for k,v in row.iteritems():
        try:
            loaded = json.loads(v)
            parsed = pd.read_json(v)
        except:
            parsed = v
        data[k] = parsed
    return data



