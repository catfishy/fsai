import time
from optparse import OptionParser

import boto3
#import fab

def runCommandsOnServer(server, commands):
    pass

def getServerList():
    return []

# start codedeploy service
def startDeployService():
    commands = []
    commands.append(['sudo','yum','-y','update'])
    commands.append(['sudo','yum','install','ruby'])
    commands.append(['sudo','yum','install','wget'])
    commands.append(['wget','https://aws-codedeploy-us-west-2.s3.amazonaws.com/latest/install','-P','/home/ec2-user/'])
    commands.append(['chmod','+x','/home/ec2-user/install'])
    commands.append(['sudo','/home/ec2-user/install','auto'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

def getDeployStatus(client, depId):
    return client.get_deployment(deploymentId=depId)['deploymentInfo']['status']

def deployCommit(commitId):
    '''
    Deploy the commit to frontend/compute servers
    '''
    client = boto3.client('codedeploy')
    revision = {'revisionType': 'GitHub',
                'gitHubLocation': {'repository': 'catfishy/fsai',
                                   'commitId': commitId}}
    # deploy to frontend
    frontend_response = client.create_deployment(applicationName='CodeDeployFSAI-App',
                                                 deploymentGroupName='frontend',
                                                 revision=revision,
                                                 deploymentConfigName='CodeDeployDefault.OneAtATime',
                                                 description='',
                                                 ignoreApplicationStopFailures=True)
    frontend_depId = frontend_response['deploymentId']
    frontend_depStatus = getDeployStatus(client, frontend_depId)
    while frontend_depStatus != 'Succeeded':
        time.sleep(3)
        frontend_depStatus = getDeployStatus(client, frontend_depId)
        if frontend_depStatus == 'Failed':
            raise Exception("Deployment %s Failed" % frontend_depId)
    
    # deploy to compute
    pass

    return frontend_depId

def sourceBashProfile():
    '''
    run 'source ${HOME}/.bash_profile' on servers
    '''
    commands = []
    commands.append(['source', '${HOME}/.bash_profile'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

def installDependencies():
    '''
    run 'python init/setup.py' on servers
    '''
    commands = []
    commands.append(['python', '${PROJECTPTH}/init/setup.py'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

def startServices():
    '''
    run init/init_services.sh on servers
    '''
    commands = []
    commands.append(['${PROJECTPTH}/init/init_services.sh'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

def stopServices():
    commands = []
    commands.append(['${PROJECTPTH}/init/stop_services.sh'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

def loadDumpData():
    '''
    load dump data on compute server (after services have started)
    '''
    # create mongo admin user
    # create mongo backup user
    # restore latest mongo dump
    pass

if __name__ == "__main__":
    parser = OptionParser()
    parser.add_option("-c", "--commit", dest="commitid", default=None,
                      help="Commit ID to deploy")

    (options, args) = parser.parse_args()

    commitId = options.commitid
    if commitId is None:
        raise Exception("Specify a commit to deploy: -c <commit ID>")

    # # start codedeploy service on target servers
    # startDeployService()

    # deploy code
    depId = deployCommit(commitId)
    print depId

    # # re-source bash profile on each server
    # sourceBashProfile()

    # OPTION: STOP AND INSTALL 

    # # Stop services on servers
    # stopServices()

    # # run setup.py (install dependencies) on each server
    # installDependencies()

    # # start services on servers
    # startServices()

    # OPTION: LOAD RECENT DUMP OF DATA

    # # load data
    # loadDumpData()

