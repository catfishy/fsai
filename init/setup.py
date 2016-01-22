'''
Make sure all necessary packages are installed
'''
import os
from subprocess import check_call, CalledProcessError

PIP_PACKAGES = ['simplejson==3.8.0',
                'beautifulsoup4==4.3.2',
                'requests==2.5.1',
                'pymongo==3.2',
                'gpy==0.6.0',
                'theano==0.6.0',
                'Pillow==2.7.0',
                'celery==3.1.19',
                'kombu==3.0.30',
                'boto3==1.2.3',
                'redis==2.10.5',
                'fabric==1.4.0']
DJANGO_PACKAGES = ['django==1.9.1',
                   'dj-database-url==0.3.0',
                   'dj-static==0.0.6',
                   'django-extensions==1.6.1',
                   'djangorestframework==3.3.2',
                   'django-compressor==1.6',
                   'drf-nested-routers==0.11.1']
SCIPY_PACKAGES = ['numpy',
                  'scipy',
                  'matplotlib==1.5.0',
                  'pandas',
                  'scikit-learn']
APACHE_PACKAGES = ['python27-devel',
                   'gcc',
                   'gcc-c++',
                   'subversion',
                   'git',
                   'httpd',
                   'make',
                   'uuid',
                   'libuuid-devel',
                   'httpd-devel',
                   'python27-imaging',
                   'mysql',
                   'boost',
                   'boost-devel']

def run_commands(cmds):
    for cmd in cmds:
        check_call(cmd)

def make_str(cmd_list):
    return ' '.join(cmd_list)

def install_apache(initenv):
    if initenv == 'FRONTEND':
        install_yum_pkgs(APACHE_PACKAGES)
        # install mod_wsgi
        proj_path = os.environ['PROJECTPTH']
        mod_wsgi_install_script = ['sudo','%s/init/install_mod_wsgi_amazon_linux.sh' % proj_path]
        check_call(mod_wsgi_install_script)
        # move config file
        cmd = ['sudo', 'cp', '%s/init/httpd.conf' % proj_path, '/etc/httpd/conf/httpd.conf']
        check_call(cmd)
        # fix permissions
        cmds = []
        cmds.append(['sudo','usermod','-a','-G','ec2-user','apache'])
        cmds.append(['chmod','710','/home/ec2-user'])
        cmds.append(['sudo','chown',':apache',proj_path])
        cmds.append(['sudo','chown','-R','root:apache','/var/log/httpd'])
        cmds.append(['sudo','chmod','-R','755','/var/log/httpd'])
        run_commands(cmds)

def install_scipy(initenv):
    PREREQS = ['gcc-c++','python27-devel','atlas-sse3-devel','lapack-devel','libpng-devel','freetype-devel']
    if initenv in ['COMPUTE', 'FRONTEND']:
        commands = []
        commands.append(['sudo','/bin/dd','if=/dev/zero','of=/var/swap.1','bs=1M','count=1024'])
        commands.append(['sudo','/sbin/mkswap','/var/swap.1'])
        commands.append(['sudo','/sbin/swapon','/var/swap.1'])
        run_commands(commands)
        install_yum_pkgs(PREREQS)
        install_pip_pkgs(SCIPY_PACKAGES,mode=None)
        commands = []
        commands.append(['sudo','swapoff','/var/swap.1'])
        commands.append(['sudo','rm','/var/swap.1'])
        run_commands(commands)
    elif initenv == 'DEV':
        cmd = ['brew','install'] + PREREQS
        check_call(cmd)
        install_pip_pkgs(SCIPY_PACKAGES,mode=None)

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
        # install
        install_yum_pkgs(['mongodb-org']) # requires manually created repo file /etc/yum.repos.d/mongodb-org-3.2.repo
        # copy conf
        cmd = ['sudo', 'cp', '%s/init/mongod.conf' % proj_path, '/etc/mongod.conf']
        check_call(cmd)
    elif initenv == 'DEV':
        cmd = ['sudo', 'brew', 'install', 'mongodb=3.2.0']
        check_call(cmd)
    pass

def install_yum_pkgs(yum_pkgs):
    for pkg in yum_pkgs:
        try:
            yum_cmd = ['sudo', 'yum', 'install', '-y', pkg]
            check_call(yum_cmd)
        except CalledProcessError:
            print "CalledProcessError: YUM Command %s returned non-zero exit status" % make_str(yum_cmd)

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

def update(initenv):
    if initenv in ['COMPUTE','FRONTEND']:
        cmd = ['sudo','yum','-y','update']
        check_call(cmd)
    elif initenv == 'DEV':
        cmd = ['sudo', 'brew', 'update']
        check_call(cmd)


def installAll():
    '''
    Install stack
    '''
    initenv = os.environ['INITENV']
    update(initenv)
    install_apache(initenv)
    install_redis(initenv)
    install_mongo(initenv)
    install_scipy(initenv)
    install_pip_pkgs(DJANGO_PACKAGES, mode=None)
    install_pip_pkgs(PIP_PACKAGES, mode=None)

if __name__ == "__main__":
    installAll()

