#!/bin/bash 

USERTYPE=$(whoami)
if [ "$USERTYPE" != "root" ]
then
    echo "Run as root"
    exit 1
fi

# Add mongodb functions to path
# Assuming mongo executables at location below
PTH='/Applications/mongodb-osx-x86_64-2.6.7/bin'
CMD='export PATH=${PATH}:'
CMD=$CMD$PTH
if grep -Fxq "$CMD" ${HOME}/.bash_profile
then
    echo "mongo path export found"
else
    echo "mongo path export not found"
    echo $CMD >> ${HOME}/.bash_profile
fi

# reload bash profile
source ${HOME}/.bash_profile

# if in compute,
# start mongodb (db=/data/db) - load most recent dump from s3
# start redis-server (conf=$PROJECTPTH'/init/redis.conf') - load most recent dump from s3
# load crontab
if [ "$INITENV" == 'COMPUTE']
then
fi

# if in prod,
# connect mongo to compute mongo
# connect redis to compute redis
# start server
if [ "$INITENV" == 'FRONTEND']
then
fi

# If in dev,
# start local mongo + redis (load most recent dump from s3)
if [ "$INITENV" == 'DEV' ]
then
    # check if already run mongod
    PRIORMONGOD=$(pgrep mongod)
    if [ "$PRIORMONGOD" == "" ]
    then
        # make sure /data/db directory exists
        echo "Starting local mongod at localhost:27017, db=/data/db"
        mkdir -p /data/db
        mongod --fork --logpath /var/log/mongodb.log
    else
        echo "Mongod already running PID: $PRIORMONGOD"
    fi
    REDISCONF=$PROJECTPTH'/init/redis.conf'
    echo "Starting local redis server with conf $REDISCONF"
    /usr/local/bin/redis-server $REDISCONF
fi

exit 0