#!/bin/bash 

# create mongo directories
sudo mkdir -p /var/log/mongodb
sudo mkdir -p /var/run/mongodb
sudo mkdir -p /var/lib/mongodb

# if in compute,
if [ "$INITENV" == 'COMPUTE']
then
    # (re)start mongod and redis
    sudo service mongod start
    sudo service redis-server start
    # load crontab
fi

# if in frontend,
if [ "$INITENV" == 'FRONTEND']
then
    # (re)start apache + mod_wsgi
    sudo service httpd start
    sudo chkconfig httpd on
fi

# If in dev,
if [ "$INITENV" == 'DEV' ]
then
    # reload bash profile
    source ${HOME}/.bash_profile

    # check if already running local mongod
    PRIORMONGOD=$(pgrep mongod)
    if [ "$PRIORMONGOD" == "" ]
    then
        # if not, start local mongod
        echo "Starting local mongod"
        mkdir -p /data/db
        mongod --config ${PROJECTPTH}'/init/mongod.conf'
    else
        echo "Mongod already running PID: $PRIORMONGOD"
    fi

    # check if already running local redis
    PRIORREDIS=$(pgrep redis)
    if [ "$PRIORREDIS" == ""]
    then:
        # if not, start local redis
        REDISCONF=$PROJECTPTH'/init/redis.conf'
        echo "Starting local redis server with conf $REDISCONF"
        /usr/local/bin/redis-server $REDISCONF
    else:
        echo "Redis already running PID: $PRIORREDIS"
    fi
fi

exit 0