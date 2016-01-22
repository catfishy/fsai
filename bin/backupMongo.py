from subprocess import check_call
import os
from datetime import datetime

import boto3
from init.config import get_config
from init.load_data import BACKUP_DIR, BACKUP_MONGO_PREFIX, BACKUP_TS_FORMAT

def backupMongoDB():
	# YYYY-mm-DD-HHMMSS
	ts = datetime.now().strftime(BACKUP_TS_FORMAT)
	output_folder = '%s_%s' % (BACKUP_MONGO_PREFIX,ts)
	output_tar = '%s.tgz' % (output_folder)
	
	# get backup user and s3 info
	config = get_config()
	mongo_user = config['MONGO_BACKUP_USER']
	mongo_pw = config['MONGO_BACKUP_PW']
	aws_email = config["AWS_EMAIL"]
	s3_bucket = config["S3_MONGO_BACKUP_BUCKET"]
	
	# backup file names
	backup_folder = os.path.join(BACKUP_DIR,output_folder)
	backup_tar = os.path.join(BACKUP_DIR,output_tar)
	
	# backup commands
	check_call(['mkdir','-p',BACKUP_DIR])
	check_call(['mongodump','-u',mongo_user,'-p',mongo_pw,'--out=%s' % backup_folder])
	check_call(['tar', '-czf', backup_tar, '-C', BACKUP_DIR, output_folder])

	# Upload to s3
	check_call(['aws','s3','cp',backup_tar,'s3://%s/' % s3_bucket,'--grants','read=emailaddress=%s' % aws_email])

	# remove local folder and tar
	check_call(['rm', '-rf', backup_folder])
	check_call(['rm', backup_tar])

	return True

if __name__ == "__main__":
	backupMongoDB()

