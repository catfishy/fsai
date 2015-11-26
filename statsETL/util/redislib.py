import sys
import simplejson as json
import redis
import socket

CACHE_SERVER = '127.0.0.1'
CACHE_PORT = 6379

def getRedisCache():
    """Get REDIS connection. Redis client is thread-safe."""
    return redis.StrictRedis(CACHE_SERVER, CACHE_PORT)


class RedisSemaphore:

    """ 
    A semaphore backed by Redis. 
    """

    def __init__(self, lock_name, connection=None):
        self.connection = connection if connection else getRedisCache()
        self.name = "lock:" + lock_name

    def lock(self, ttl=None):
        """ 
        Create the lock in redis. Lock TTL is optional. 
        """
        if self.isLocked():
            if ttl:
                return self.connection.expire(self.name, ttl)
            raise Exception("Lock already exists.")
        if not self.connection.setnx(self.name, socket.gethostname()):
            return False
        # Caution: If both the expiration and unlock fail, the lock will
        # incorrectly persist forever.
        if ttl and type(ttl) is int:
            if not self.connection.expire(self.name, ttl):
                self.unlock()
                return False
        return True

    def unlock(self, override=False):
        """ 
        Release the lock in redis.
        This machine must be the machine that created the lock in order to
        release it (identified by hostname).
        Optional: override this requirement.
        """
        if not override:
            val = self.connection.get(self.name)
            if not val:
                return True
            elif val != socket.gethostname():
                raise Exception("Only a lock's owner may release that lock.")
        return self.connection.delete(self.name)

    def isLocked(self):
        """ 
        Test if a lock exists (in other words: "is locked."). 
        """
        return (None != self.connection.get(self.name))


class RedisTTLCachePolicy:
    """ 
    Specify a cache policy
    Potentially variable by object being cached
    """
    def __init__(self, default_ttl=3600):
        self.default_ttl = 3600

    def getTTL(self, object_being_cached):
        return self.default_ttl


class RedisTTLCache(dict):
    """ 
    A Redis cache that uses a predefined TTL or TTL policy to cache
    objects and retrieve them.

    WARNING: Mostly thread safe. There may be unnecessary writes or overwritten
    TTLs when two threads try to cache something simultaneously, but nothing
    seriously destructive. There is the issue of two threads fetching the
    same item from the cache, making separate and conflicting modifications,
    and then racing to see who saves first, but that's a fundamental limitation
    of non-transactional datastores rather than an issue specific to this code.

    See https://github.com/andymccurdy/redis-py#thread-safety

    WARNING: NOT A PROPER DICT SUBCLASS. 

    """
    def __init__(self, cache_name, ttl_policy=None, connection=None):
        self.connection = connection if connection else getRedisCache()
        if ttl_policy and isinstance(ttl_policy, RedisTTLCachePolicy):
            self.ttl_policy = ttl_policy
        else:
            self.ttl_policy = None
        self.cache_name = cache_name

    def _getKeyName(self, key):
        return self.cache_name + ":" + str(key)

    def __getitem__(self, key):
        """ Grab an item from Redis. """
        # WARNING: "x is in RedisTTLCache" operations are not safe and may lead
        # to a race condition. Don't use them.
        val = self.connection.get(self._getKeyName(key))
        if val:
            return json.loads(val)
        # This technically breaks the dict subclassing, should return a KeyError.
        return None

    def __setitem__(self, key, val):
        """ Add an item into Redis. """
        # WARNING: May overwrite an item.
        key = self._getKeyName(key)
        self.connection.set(key, json.dumps(val))
        if self.ttl_policy is not None:
            self.connection.expire(key, self.ttl_policy.getTTL(val))

    def __delitem__(self, key):
        """ 
        Deleting a key evicts it from the cache 
        """
        self.connection.delete(self._getKeyName(key))

    def __contains__(self, key):
        """ A lazy implementation of the dict contains function."""
        if self.__getitem__(key):
            return True
        return False


class RedisHashCache(RedisTTLCache):
    """Much the same as the RedisTTLHash, but overrides the get and set
       methods to use the redis hash type. However, does not overwrite
       already existing keys."""
    def __getitem__(self, key):
        """ Grab an item from Redis. """
        val = self.connection.hget(self.cache_name, key)
        if val:
            return json.loads(val)
        raise KeyError('Key %s not in cache' % str(key))

    def __setitem__(self, key, val):
        """ Add an item into Redis. """
        # WARNING: Will NOT overwrite an item.
        self.connection.hsetnx(self.cache_name, key, json.dumps(val))

    def __contains__(self, key):
        if self.connection.hget(self.cache_name, key):
            return True
        return False

    def keys(self):
        return self.connection.keys()

if __name__ == "__main__":
    cache = RedisTTLCache("foobartest")
    print "setting"
    cache['banana'] = 'phone'
    print "checking"
    print ('banana' in cache)
    print "getting"
    print cache['banana']
    print "deleting"
    del cache['banana']
    print "getting non-existent item"
    print cache['banana']
