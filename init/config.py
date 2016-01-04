'''
Load the correct configurations based on INITENV
'''
import os
import simplejson

def get_config():
	initenv = os.environ['INITENV']
	config_file = os.path.join(os.environ['PROJECTPTH'], 'init', 'config.json')
	with open(config_file) as infile:
		raw_config = simplejson.load(infile)
	parsed_config = {}
	for k,v in raw_config.iteritems():
		if isinstance(v, dict):
			parsed_config[k] = v.get(initenv,None)
		else:
			parsed_config[k] = v
	return parsed_config