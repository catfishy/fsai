#!/bin/bash 

# if in compute,
if [ "$INITENV" == 'COMPUTE']
then
    # stop any python jobs
    sudo pkill python
    # backup mongo and redis
    python ${PROJECTPTH}'bin/backupMongo.py'
    python ${PROJECTPTH}'bin/backupRedis.py'
    # stop mongod and redis
    sudo service mongod stop
    sudo service redis-server stop
fi

# if in frontend,
if [ "$INITENV" == 'FRONTEND']
then
    # stop apache
    sudo service httpd stop
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
        echo "No prior MONGOD"
    else
        kill $PRIORMONGOD
        echo "Mongod killed: $PRIORMONGOD"
    fi

    # check if already running local redis
    PRIORREDIS=$(pgrep redis)
    if [ "$PRIORREDIS" == ""]
    then:
        echo "No prior REDIS PID"
    else:
        kill $PRIORREDIS
        echo "Redis killed: $PRIORREDIS"
    fi
fi

exit 0