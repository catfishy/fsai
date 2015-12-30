#!/bin/bash 

USERTYPE=$(whoami)
if [ "$USERTYPE" != "root" ]
then
    echo "Run as root"
    exit 1
fi

# dev or production
if [ "$1" == "dev" ]
then
    INITENV='DEV'
elif [ "$1" == "prod" ]
then
    INITENV='PROD'
else
    echo "Specify 'dev' or 'prod' as first argument"
    exit 1
fi

# Add project home to python path
PROJECTPTH='/usr/local/fsai'
CMD='export PYTHONPATH=${PYTHONPATH}:'
CMD=$CMD$PROJECTPTH
if grep -Fxq "$CMD" ${HOME}/.bash_profile
then
    echo "python path export found"
else
    echo "python path export not found"
    echo $CMD >> ${HOME}/.bash_profile
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

# If in dev, 
# start mongodb with db=/data/db directory
# start redis-server with conf=/usr/local/etc/redis.conf

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
    # check if already run redis
    REDISCONF=$PROJECTPTH'/init/redis.conf'
    echo "Starting local redis server with conf $REDISCONF"
    /usr/local/bin/redis-server $REDISCONF
fi

# TODO: call python init script to install packages and start daemons

exit 0