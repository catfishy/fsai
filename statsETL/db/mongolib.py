import sys
import simplejson as json
import pandas as pd

from pymongo import MongoClient
from bson.objectid import ObjectId

from init.config import get_config


class MongoConn:

    def __init__(self, db=None):
        raise Exception("USE CHILD CLASS")

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

    def ensureIndex(self, coll, index_args, unique=False, sparse=False):
        index = coll.create_index(index_args, unique=unique, sparse=sparse)
        return index

    def bulkInsert(self, coll, docs):
        if type(docs) != list:
            raise Exception("Documents arg must be a list of dicts")
        doc_ids = coll.insert(docs)
        return doc_ids

    def parallelScan(self, coll, num_cursors=1):
        cursors = coll.parallel_scan(num_cursors)
        return cursors

class NBAMongoConn(MongoConn):

    def __init__(self):
        # create mongo client
        config = get_config()
        MONGOD_HOST = config['MONGOD_HOST']
        MONGOD_PORT = config['MONGOD_PORT']
        NBA_USER = config["MONGO_NBA_USER"]
        NBA_PW = config["MONGO_NBA_PW"]
        NBA_DB = config["MONGO_NBA_DB"]
        uri = "mongodb://%s:%s@%s:%s/%s?authMechanism=SCRAM-SHA-1" % (NBA_USER, NBA_PW, MONGOD_HOST, MONGOD_PORT, NBA_DB)
        self.client = MongoClient(uri)
        self.db = self.client[NBA_DB]

# get nba db conn
nba_conn = NBAMongoConn()

# raw stats
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

# vector outputs
nba_player_outputs_collection = nba_conn.getCollection("nba_player_outputs")
nba_team_outputs_collection = nba_conn.getCollection("nba_team_outputs")

# util tables
nba_stat_ranges_collection = nba_conn.getCollection("nba_stat_ranges")

# indices
nba_conn.ensureIndex(nba_teams_collection, [("team_id", 1),("season", 1)], unique=True)
nba_conn.ensureIndex(depth_collection, [("team_id", 1),("season", 1)], unique=True)
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
nba_conn.ensureIndex(nba_player_outputs_collection, [("game_id", 1),("player_id", 1), ("window", 1)], unique=True)
nba_conn.ensureIndex(nba_team_outputs_collection, [("game_id", 1),("team_id", 1), ("window", 1)], unique=True)


