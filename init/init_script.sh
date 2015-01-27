PROJECT_PATH='/usr/local/fsai'
EXPORT_CMD='export PYTHONPATH=${PYTHONPATH}:$PROJECT_PATH'
echo 'export PYTHONPATH=${PYTHONPATH}:/usr/local/fsai' >> ${HOME}/.bash_profile
source ${HOME}/.bash_profile