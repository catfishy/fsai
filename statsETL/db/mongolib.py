import sys

from pymongo import MongoClient
from bson.objectid import ObjectId

# TODO: start mongod
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

    def updateDocument(self, coll, doc):
        doc_id = coll.update(doc)
        return doc_id

    def saveDocument(self, coll, doc):
        '''
        updates if doc found, else inserts
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

    def ensureIndex(self, coll, index_args, unique=False):
        index = coll.create_index(index_args, unique=unique)
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

# get collections
team_game_collection = nba_conn.getCollection("team_games")
player_game_collection = nba_conn.getCollection("player_games")
game_collection = nba_conn.getCollection("games")
team_collection = nba_conn.getCollection("teams")
player_collection = nba_conn.getCollection("players")
upcoming_collection = nba_conn.getCollection("upcoming")