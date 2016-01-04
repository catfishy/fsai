'''
Make sure all necessary packages are installed
'''
import os
from subprocess import check_call, CalledProcessError

YUM_PACKAGES = ['nginx']
PIP_PACKAGES = ['simplejson',
                'beautifulsoup4==4.3.2',
                'requests==2.5.1',
                'pymongo==2.7.2',
                'gpy==0.6.0',
                'theano==0.6.0',
                'Pillow==2.7.0',
                'celery==3.1.19',
                'kombu==3.0.30',
                'boto',
                'django==1.9.1',
                'dj-database-url==0.3.0',
                'django-extensions==1.6.1',
                'djangorestframework==3.3.2',
                'django-compressor==1.6',
                'drf-nested-routers==0.11.1',
                'redis==2.10.5']
SCIPY_PACKAGES = ['numpy',
                  'scipy',
                  'matplotlib==1.5.0',
                  'pandas',
                  'scikit-learn']


def make_str(cmd_list):
    return ' '.join(cmd_list)

def install_scipy(initenv):
    commands = []
    if initenv in ['COMPUTE', 'FRONTEND']:
        commands.append(['sudo','/bin/dd','if=/dev/zero','of=/var/swap.1','bs=1M','count=1024'])
        commands.append(['sudo','/sbin/mkswap','/var/swap.1'])
        commands.append(['sudo','/sbin/swapon','/var/swap.1'])
        commands.append(['sudo','yum','-y','install','gcc-c++','python27-devel','atlas-sse3-devel','lapack-devel','libpng-devel','freetype-devel'])
        install_pip_pkgs(SCIPY_PACKAGES,mode=None)
        commands.append(['sudo','swapoff','/var/swap.1'])
        commands.append(['sudo','rm','/var/swap.1'])
    elif initenv == 'DEV':
        commands.append(['brew','install','gcc-c++','python27-devel','atlas-sse3-devel','lapack-devel','libpng-devel','freetype-devel'])
        install_pip_pkgs(SCIPY_PACKAGES,mode=None)
    for cmd in commands:
        check_call(cmd)


def install_redis(initenv):
    if initenv in ['COMPUTE','FRONTEND']:
        proj_path = os.environ['PROJECTPTH']
        redis_install_script = ['%s/init/install_redis_amazon_linux.sh' % proj_path]
        check_call(redis_install_script)
    elif initenv == 'DEV':
        cmd = ['sudo', 'brew', 'install', 'redis=3.0.5']
        check_call(cmd)
    pass

def install_mongo(initenv):
    if initenv in ['COMPUTE','FRONTEND']:
        # configure package managment system
        proj_path = os.environ['PROJECTPTH']
        repo_copy = ['sudo', 'cp', '%s/init/mongodb-org-3.2.repo' % proj_path, '/etc/yum.repos.d/']
        check_call(repo_copy)
        cmd = ['sudo', 'yum', 'install', '-y', 'mongodb-org'] # requires manually created repo file /etc/yum.repos.d/mongodb-org-3.2.repo
        check_call(cmd)
    elif initenv == 'DEV':
        cmd = ['sudo', 'brew', 'install', 'mongodb']
        check_call(cmd)
    pass

def install_pip_pkgs(pip_pkgs, mode=None):
    for pkg in pip_pkgs:
        try:
            if mode == "upgrade":
                pip_cmd = ['sudo', 'pip', 'install', '--upgrade', pkg]
            else:
                pip_cmd = ['sudo', 'pip', 'install', pkg]
            check_call(pip_cmd)
        except CalledProcessError:
            print "CalledProcessError: PIP Command %s returned non-zero exit status" % make_str(pip_cmd)

def installAll():
    initenv = os.environ['INITENV']
    install_mongo(initenv)
    install_redis(initenv)
    install_scipy(initenv)
    install_pip_pkgs(PIP_PACKAGES, mode=None)

if __name__ == "__main__":
    installAll()

