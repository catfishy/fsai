'''
Make sure all necessary packages are installed
'''
import os
from subprocess import check_call, CalledProcessError

PIP_PACKAGES = ['simplejson',
                'beautifulsoup4==4.3.2',
                'requests==2.5.1',
                'pymongo==2.7.2',
                'scikit-learn==0.15.2',
                'gpy==0.6.0',
                'matplotlib',
                'theano==0.6.0',
                'Pillow==2.7.0',
                'celery==3.1.19',
                'kombu==3.0.30',
                'pandas==0.16.2']

def make_str(cmd_list):
    return ' '.join(cmd_list)

def install_redis(initenv):
    if initenv in ['COMPUTE','FRONTEND']:
        proj_path = os.environ['PROJECTPTH']
        redis_install_script = ['%s/init/install_redis_amazon_linux.sh' % proj_path]
        check_call(redis_install_script)
    elif initenv == 'DEV':
        cmd = ['brew', 'install', 'redis=3.0.5']
        check_call(cmd)
    pass

def install_mongo(initenv):
    if initenv in ['COMPUTE','FRONTEND']:
        # configure package managment system
        proj_path = os.environ['PROJECTPTH']
        repo_copy = ['cp', '%s/init/mongodb-org-3.2.repo' % proj_path, '/etc/yum.repos.d/']
        check_call(repo_copy)
        cmd = ['yum', 'install', '-y', 'mongodb-org'] # requires manually created repo file /etc/yum.repos.d/mongodb-org-3.2.repo
        check_call(cmd)
    elif initenv == 'DEV':
        cmd = ['brew', 'install', 'mongodb']
        check_call(cmd)
    pass

def install_pip_pkgs(pip_pkgs, mode=None):
    for pkg in pip_pkgs:
        try:
            if mode == "upgrade":
                pip_cmd = ['pip', 'install', '--upgrade', pkg]
            else:
                pip_cmd = ['pip', 'install', pkg]
            check_call(pip_cmd)
        except CalledProcessError:
            print "CalledProcessError: PIP Command %s returned non-zero exit status" % make_str(pip_cmd)

if __name__ == "__main__":
    # run as sudo
    # check INITENV
    # if compute/frontend, install yum/pip
    # if dev, install pip/brew

