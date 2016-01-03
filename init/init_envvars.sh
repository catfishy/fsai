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

CMD='export INITENV='
# remove old setting
grep -v "$CMD" ${HOME}/.bash_profile > ${HOME}/.bash_profile_cleaned
mv ${HOME}/.bash_profile_cleaned ${HOME}/.bash_profile
echo "Setting init env"
echo $CMD\'$INITENV\' >> ${HOME}/.bash_profile

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

# reload bash profile
source ${HOME}/.bash_profile
