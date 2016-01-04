import boto3
import fab

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

def deployCommit(commitId):
    '''
    Deploy the commit to frontend/compute servers
    '''
    client = boto3.client('codedeploy')
    frontend_depgrp = client.get_deployment_group(applicationName='CodeDeployFSAI-App', deploymentGroupName='frontend')
    revision = {'revisionType': 'GitHub',
                'gitHubLocation': {'repository': 'catfishy/fsai',
                                   'commitId': commitId}}
    frontend_response = client.create_deployment(applicationName='CodeDeployFSAI-App',
                                                 deploymentGroupName='frontend',
                                                 revision=revision,
                                                 deploymentConfigName='CodeDeployDefault.OneAtATime',
                                                 description='',
                                                 ignoreApplicationStopFailures=True)
    frontend_depId = frontend_response['deploymentId']
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

def startServers():
    '''
    run init/init_servers.sh on servers
    '''
    commands = []
    commands.append(['${PROJECTPTH}/init/init_servers.sh'])
    for server in getServerList():
        runCommandsOnServer(server, commands)

if __name__ == "__main__":
    commitId = ''

    # # start codedeploy service on target servers
    # startDeployService()

    # deploy code
    depId = deployCommit(commitId)
    print depId

    # # re-source bash profile on each server
    # sourceBashProfile()

    # # run setup.py (install dependencies) on each server
    # installAll()

    # # start services on servers
    # startServers()

