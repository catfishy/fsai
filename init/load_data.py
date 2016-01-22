import os
from subprocess import check_call, check_output, CalledProcessError
from datetime import datetime

from init.config import get_config

BACKUP_DIR = os.path.join(os.environ['HOME'],'tmp/backup/')
BACKUP_TS_FORMAT = '%Y-%m-%d-%H%M%S'
BACKUP_MONGO_PREFIX = 'mongodump'
BACKUP_REDIS_PREFIX = 'redisdump'

def restoreLatestMongoDump(initenv):
    '''
    Assuming mongod started 
    '''
    if initenv in ['COMPUTE', 'DEV']:
        # get backup user and s3 bucket
        config = get_config()
        mongo_user = config['MONGO_BACKUP_USER']
        mongo_pw = config['MONGO_BACKUP_PW']
        s3_bucket = config["S3_MONGO_BACKUP_BUCKET"]

        # find latest data dump in s3
        s3_ls_output = check_output(['aws','s3','ls','s3://fsai-mongo-backup'])
        mongodump_files = [_ for _ in s3_ls_output.split() if BACKUP_PREFIX in _]
        if len(mongodump_files) == 0:
            raise Exception("NO DATADUMPS FOUND ON S3")
        by_timestamp = {}
        for dump_file in mongodump_files:
            ts_str= dump_file.split('_')[-1].replace('.tgz','')
            ts = datetime.strptime(ts_str, BACKUP_TS_FORMAT)
            by_timestamp[ts] = dump_file
        most_recent_ts = sorted(by_timestamp.keys())[-1]
        dump_tgz_file = by_timestamp[most_recent_ts]
        print "CHOSEN DUMP: %s" % dump_tgz_file
        local_dump_file = os.path.join(BACKUP_DIR,dump_tgz_file)
        local_dump_folder = os.path.join(BACKUP_DIR,dump_tgz_file.replace('.tgz',''))

        # download latest data dump to BACKUP_DIR (going through aws cli)
        check_call(['mkdir','-p',BACKUP_DIR])
        check_call(['aws','s3','cp','s3://%s/%s' % (s3_bucket,dump_tgz_file), local_dump_file])
        check_call(['tar', '-xzf', local_dump_file, '-C', BACKUP_DIR])

        # login with admin user and load dump
        check_call(['mongorestore', '-u', mongo_user, '-p', mongo_pw, local_dump_folder])

        # remove local dumps
        check_call(['rm', '-rf', local_dump_folder])
        check_call(['rm', local_dump_file])

        return True
    else:
        return True

def restoreLatestRedisDump(initenv):
    if initenv in ['COMPUTE', 'DEV']:
        # download latest data dump (going through aws cli)
        pass

def createMongoAdminUser():
    '''
    Try logging in with locahost exception and creating admin user
    '''
    config = get_config()
    # create user admin
    user = config["MONGO_ADMIN_USER"]
    pw = config["MONGO_ADMIN_PW"]
    authdb = config["MONGO_ADMIN_DB"]
    mongo_cmd = 'db.createUser({user: "%s", pwd: "%s", roles: [ { role: "userAdminAnyDatabase", db: "%s" } ]})' % (user, pw, authdb)
    cmd = ['mongo','admin','--eval',mongo_cmd]
    check_call(cmd)

def createMongoBackupUser():
    '''
    Create the backup user (with backup and restore privileges)
    '''
    config = get_config()
    # log in with user admin and create nba db user
    admin_user = config["MONGO_ADMIN_USER"]
    admin_pw = config["MONGO_ADMIN_PW"]
    admin_db = config["MONGO_ADMIN_DB"]
    backup_user = config["MONGO_BACKUP_USER"]
    backup_pw = config["MONGO_BACKUP_PW"]
    mongo_cmd = 'db.createUser({user: "%s", pwd: "%s", roles: [ {role: "backup", db: "%s" },{role: "restore", db: "%s"} ]})' % (backup_user, backup_pw, admin_db, admin_db)
    cmd = ['mongo','-u',admin_user,'-p',admin_pw,'--authenticationDatabase', admin_db, admin_db, '--eval',mongo_cmd]
    check_call(cmd)


def createMongoNBAUser():
    '''
    Create the DB owner for the NBA table
    '''
    config = get_config()
    # log in with user admin and create nba db user
    admin_user = config["MONGO_ADMIN_USER"]
    admin_pw = config["MONGO_ADMIN_PW"]
    admin_db = config["MONGO_ADMIN_DB"]
    nba_user = config["MONGO_NBA_USER"]
    nba_pw = config["MONGO_NBA_PW"]
    nba_db = config["MONGO_NBA_DB"]
    mongo_cmd = 'db.createUser({user: "%s", pwd: "%s", roles: [ { role: "dbOwner", db: "%s" } ]})' % (nba_user, nba_pw, nba_db)
    cmd = ['mongo','-u',admin_user,'-p',admin_pw,'--authenticationDatabase', admin_db, nba_db, '--eval',mongo_cmd]
    check_call(cmd)

def createRedisAdminUser():
    pass

if __name__ == '__main__':
    initenv = os.environ['INITENV']

    restoreLatestMongoDump(initenv)



