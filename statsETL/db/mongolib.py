import sys

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
        result = coll.update_one({"_id": row_id}, {'$set' : doc}, upsert=upsert)
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
        result = coll.find_one({"_id": ObjectId(doc_id)})
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

# create collections
team_game_collection = nba_conn.getCollection("team_games")
player_game_collection = nba_conn.getCollection("player_games")
game_collection = nba_conn.getCollection("games")
team_collection = nba_conn.getCollection("teams")
player_collection = nba_conn.getCollection("players")
upcoming_collection = nba_conn.getCollection("upcoming") # for upcoming bets
future_collection = nba_conn.getCollection("future") # for future nba games
projection_collection = nba_conn.getCollection("projections") # for modeling projections by player
espn_stat_collection = nba_conn.getCollection("espnstats") # for espn team stats
espn_player_stat_collection = nba_conn.getCollection("espnplayerstats")
espn_depth_collection = nba_conn.getCollection("depthcharts")
advanced_collection = nba_conn.getCollection("advanced")
onoff_collection = nba_conn.getCollection("onoff")
two_man_collection = nba_conn.getCollection("two_man")
shot_chart_collection = nba_conn.getCollection("shot_chart")
shot_collection = nba_conn.getCollection("shot")
defense_collection = nba_conn.getCollection("defense")
rebound_collection = nba_conn.getCollection("rebound")
pass_collection = nba_conn.getCollection("pass")

# ensure indices
nba_conn.ensureIndex(team_collection, [("url", 1)])
nba_conn.ensureIndex(player_collection, [("url", 1)])
nba_conn.ensureIndex(team_game_collection, [("team_id", 1)])
nba_conn.ensureIndex(team_game_collection, [("game_id", 1)])
nba_conn.ensureIndex(player_game_collection, [("player_id", 1)])
nba_conn.ensureIndex(player_game_collection, [("game_id", 1)])
nba_conn.ensureIndex(game_collection, [('url',1)])
nba_conn.ensureIndex(team_game_collection, [("team_id", 1),("game_id", 1)], unique=True)
nba_conn.ensureIndex(player_game_collection, [("player_id", 1),("game_id", 1)], unique=True)
nba_conn.ensureIndex(projection_collection, [("player_id", 1),("time", 1)], unique=True)
nba_conn.ensureIndex(espn_stat_collection, [('time', 1)], unique=True)
nba_conn.ensureIndex(espn_player_stat_collection, [("player_id", 1),('time', 1)], unique=True)
nba_conn.ensureIndex(espn_depth_collection, [('time', 1)], unique=True)
nba_conn.ensureIndex(advanced_collection, [("player_id", 1),('time', 1),('team_id', 1)], unique=True)
nba_conn.ensureIndex(onoff_collection, [("player_id", 1),('time', 1),('team_id', 1)], unique=True)
nba_conn.ensureIndex(two_man_collection, [("player_one", 1),("player_two", 1),('time', 1),('team_id', 1)], unique=True, sparse=True)
nba_conn.ensureIndex(shot_chart_collection, [("player_id", 1),("game_id", 1),("time", 1)], unique=True)
nba_conn.ensureIndex(shot_collection, [("player_id", 1),("game_id", 1),("time", 1)], unique=True)
nba_conn.ensureIndex(defense_collection, [("player_id", 1),("game_id", 1),("year", 1)], unique=True)
nba_conn.ensureIndex(rebound_collection, [("player_id", 1),("game_id", 1),("year", 1)], unique=True)
nba_conn.ensureIndex(pass_collection, [("player_id", 1),("game_id", 1),("year", 1)], unique=True)

def convertESPNPlayerID(PID):
    '''
    Converts espn player id to BR player id
    '''
    row = player_collection.find_one({'espn_id': PID})
    return row['_id']



