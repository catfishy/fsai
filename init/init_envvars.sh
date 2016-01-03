#!/bin/bash 

# dev or production
if [ "$DEPLOYMENT_GROUP_NAME" == "frontend" ]
then
    INITENV='FRONTEND'
elif [ "$DEPLOYMENT_GROUP_NAME" == "compute" ]
then
    INITENV='COMPUTE'
else
    INITENV='DEV'
fi

# Add initenv
CMD='export INITENV='
grep -v "$CMD" ${HOME}/.bash_profile > ${HOME}/.bash_profile_cleaned
mv ${HOME}/.bash_profile_cleaned ${HOME}/.bash_profile
echo "Setting init env"
echo $CMD\'$INITENV\' >> ${HOME}/.bash_profile

# Add project path
CMD='export PROJECTPTH='
PROJECTPTH='/usr/local/fsai'
grep -v "$CMD" ${HOME}/.bash_profile > ${HOME}/.bash_profile_cleaned
mv ${HOME}/.bash_profile_cleaned ${HOME}/.bash_profile
echo "Setting project path"
echo $CMD\'$PROJECTPTH\' >> ${HOME}/.bash_profile

if [ "${PYTHONPATH/$PROJECTPTH}" = "$PYTHONPATH" ] ; then
    CMD='export PYTHONPATH=${PYTHONPATH}:'
    CMD=$CMD$PROJECTPTH
    if grep -Fxq "$CMD" ${HOME}/.bash_profile
    then
        echo "python path export found"
    else
        echo "python path export not found"
        echo $CMD >> ${HOME}/.bash_profile
    fi
else
    echo "Project path already in python path"
fi

# reload bash profile
. ${HOME}/.bash_profile
exit 0
