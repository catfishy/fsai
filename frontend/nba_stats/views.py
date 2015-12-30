from datetime import datetime, timedelta
import json
from collections import defaultdict
import re

from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response
import pandas as pd

from statsETL.bball.apiExtraction import *
from statsETL.bball.statsExtraction import getTeamsForDay, getPlayersForDay, getPlayerVector, getTeamVector
from statsETL.util.crawler import daterange

class DailyPlayerVectors(views.APIView):

    def fromCacheOrExtract(self, arg, types_left):
        if isinstance(types_left, list):
            types_left = set(types_left)
        data_row = {}
        # look in cache
        cached_response = PlayerAPIExtractor.getAPIResponseFromCache(arg)
        if cached_response is not None:
            for t,v in cached_response.iteritems():
                if t in types_left:
                    data_row.update(v)
                    types_left.discard(t)
        else:
            print "NO CACHED RESPONSE FOR %s" % (arg,)

        # extract if necessary
        if len(types_left) > 0:
            pid, tid, gid = arg
            vector = getPlayerVector(pid, tid, gid, recalculate=False)
            row = PlayerAPIExtractor().extractVectors(vector, list(types_left), cache=True)
            data_row.update(row)
        else:
            print "All keys found in cache"

        # remove row if empty windowed stats
        relevant_keys = [_ for _ in data_row.keys() if 'MIN' in _]
        if len(relevant_keys) > 0 and all([not data_row[_] for _ in relevant_keys]):
            data_row = None

        return data_row

    def generateResponse(self, row_dict, request_type, color):
        if request_type == 'basic':
            header_keys = [_[0] for _ in BASIC_STAT_TYPE_KEYS]
            headers = [{'key':cleanKey(_[0]),'name':_[1]} for _ in BASIC_STAT_TYPE_KEYS]
            # create position row lists
            rows = {'G': [], 'F': [], 'C': []}
            for (pid_key, tid_key, gid_key), row in row_dict.iteritems():
                filtered_row = {_:None for _ in header_keys}
                translations = {}
                for k in row.keys():
                    prefix, suffix = APIExtractor.splitKey(k)
                    if suffix in filtered_row:
                        filtered_row[suffix] = row[k]
                        translations[suffix] = k
                # get position
                pos = filtered_row['positions'] # get values for position
                if pos is None:
                    continue
                # add pid/gid/tid keys to row
                key_update = {'pid_key': pid_key, 'tid_key': tid_key, 'gid_key': gid_key}
                # put in corresponding positional list
                for pos_bin in [_ for _ in ['G','F','C'] if _ in pos]:
                    colored_row = {cleanKey(k): {'v': v, 'c': color.extractColor(pos_bin,translations[k],v)} for k,v in filtered_row.iteritems()}
                    colored_row.update(key_update)
                    rows[pos_bin].append(colored_row)

            data = {'headers': headers,
                    'rows': rows}
        elif request_type == 'matchup':
            header_keys = [_[0] for _ in MATCHUP_STAT_TYPE_KEYS]
            header_names = {cleanKey(k):(v,i) for i,(k,v) in enumerate(MATCHUP_STAT_TYPE_KEYS)}
            flipped_header_keys = ['stat_key'] + PLAYER_STAT_TYPE['matchup']
            headers = [{'key': _, 'name': STAT_TYPE_NAMES[_]} for _ in flipped_header_keys]

            whole_row = row_dict.values()[0] 
            gid = whole_row['gid']
            pid = whole_row['pid']
            # create rows out of each data type
            by_stattype = {}
            for k in whole_row.keys():
                prefix, suffix = APIExtractor.splitKey(k)
                if suffix in header_keys:
                    if prefix not in by_stattype:
                        by_stattype[prefix] = {_:None for _ in header_keys}
                    by_stattype[prefix][suffix] = whole_row[k]
            all_rows = []
            print by_stattype.keys()
            for prefix, row in by_stattype.iteritems():
                colored_row = {cleanKey(k): {'v': v, 'c': color.extractColor('ALL','%s_%s' % (prefix,k),v)} for k,v in row.iteritems()}
                colored_row['stat_key'] = prefix
                all_rows.append(colored_row)
            
            # flip it
            rows_df = pd.DataFrame(all_rows)
            rows_df.set_index('stat_key',inplace=True)
            flipped_rows = {}
            for k,v in rows_df.to_dict('dict').iteritems():
                header_name, header_index = header_names[k]
                new_row = {'stat_key': {'v': header_name, 'c': 'transparent'}}
                new_row.update(v)
                flipped_rows[header_index] = new_row
            flipped_sorted_rows = [flipped_rows[_] for _ in sorted(flipped_rows.keys())]

            rows = {'ALL': flipped_sorted_rows}
            data = {'headers': headers,
                    'rows': rows,
                    'gid': gid,
                    'pid': pid}
        else:
            raise Exception("Invalid request type")
        return data

    def get(self, request):
        params = self.request.GET

        # generate stats query
        try:
            request_type = params['type']
            if request_type == 'matchup':
                arg_pid = params['pid']
                arg_gid = params['gid']
                arg_tid = params['tid']
                color = colorExtractor('player', gid=arg_gid)
            elif request_type == 'basic':
                from_date = datetime.strptime(params['from'],'%Y-%m-%d')
                to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
                color = colorExtractor('player', date=to_date)
            else:
                raise Exception("Invalid Type specified")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)

        types = PLAYER_STAT_TYPE[request_type]
        row_dict = {}
        args = []
        
        # query and load stats
        if request_type == 'basic':
            for date in daterange(from_date, to_date):
                new_args = getPlayersForDay(date)
                args += new_args
        elif request_type == 'matchup':
            args.append((arg_pid, arg_tid, arg_gid))

        for arg in args:
            result = self.fromCacheOrExtract(arg, types)
            if result is not None:
                row_dict[arg] = result

        # generate response according to request type
        data = self.generateResponse(row_dict, request_type, color)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)

'''
Return precalculated NBA team stat vectors in JSON format
'''

class DailyTeamVectors(views.APIView):

    def fromCacheOrExtract(self, arg, types_left):
        if isinstance(types_left, list):
            types_left = set(types_left)
        data_row = {}
        # look in cache
        cached_response = TeamAPIExtractor.getAPIResponseFromCache(arg)
        if cached_response is not None:
            for t,v in cached_response.iteritems():
                if t in types_left:
                    data_row.update(v)
                    types_left.discard(t)
        else:
            print "NO CACHED RESPONSE FOR %s" % (arg,)
        
        # extract if necessary
        if len(types_left) > 0:
            tid, gid = arg
            vector = getTeamVector(tid, gid, recalculate=False)
            row = TeamAPIExtractor().extractVectors(vector, list(types_left), cache=True)
            data_row.update(row)

        return data_row

    def generateResponse(self, row_dict, request_type, color):
        # get headers
        if request_type == 'basic':
            header_keys = [_[0] for _ in BASIC_STAT_TYPE_KEYS]
            headers = [{'key':cleanKey(_[0]),'name':_[1]} for _ in BASIC_STAT_TYPE_KEYS]
        elif request_type == 'matchup':
            header_keys = [_[0] for _ in MATCHUP_STAT_TYPE_KEYS]
            headers = [{'key':cleanKey(_[0]),'name':_[1]} for _ in MATCHUP_STAT_TYPE_KEYS]
        else:
            raise Exception("Invalid request type")
        # filter rows
        if request_type == 'basic':
            # create position row lists
            all_rows = []
            for (tid_key, gid_key), row in row_dict.iteritems():
                filtered_row = {_:None for _ in header_keys}
                translations = {}
                for k in row.keys():
                    prefix, suffix = APIExtractor.splitKey(k)
                    if suffix in filtered_row:
                        filtered_row[suffix] = row[k]
                        translations[suffix] = k
                # add color
                colored_row = {cleanKey(k): {'v': v, 'c': color.extractColor('ALL',translations[k],v)} for k,v in filtered_row.iteritems()}
                # add tid/gid keys to row
                colored_row['tid_key'] = tid_key
                colored_row['gid_key'] = gid_key
                all_rows.append(colored_row)
            rows = {'ALL': all_rows}
            data = {'headers': headers,
                    'rows': rows}
        elif request_type == 'matchup':
            whole_row = row_dict.values()[0]
            # create rows out of each data type
            by_stattype = {}
            for k in whole_row.keys():
                prefix, suffix = APIExtractor.splitKey(k)
                if suffix in header_keys:
                    if prefix not in by_stattype:
                        by_stattype[prefix] = {_:None for _ in header_keys}
                    by_stattype[prefix][suffix] = whole_row[k]
            all_rows = []
            for prefix, row in by_stattype.iteritems():
                colored_row = {cleanKey(k): {'v': v, 'c': color.extractColor('ALL','%s_%s' % (prefix,k),v)} for k,v in row.iteritems()}
                all_rows.append(colored_row)
            rows = {'ALL': all_rows}
            data = {'headers': headers,
                    'rows': rows}
        return data


    def get(self, request):
        params = self.request.GET

        # generate stats query
        try:
            request_type = params['type']
            if request_type == 'matchup':
                arg_gid = params['gid']
                arg_tid = params['tid']
                color = colorExtractor('player', gid=arg_gid)
            elif request_type == 'basic':
                from_date = datetime.strptime(params['from'],'%Y-%m-%d')
                to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
                color = colorExtractor('team', date=to_date)
            else:
                raise Exception("Invalid Type specified")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)

        types = TEAM_STAT_TYPE[request_type]
        row_dict = {}
        args = []
        
        # query and load stats
        if request_type == 'basic':
            for date in daterange(from_date, to_date):
                new_args = getTeamsForDay(date)
                args += new_args
        elif request_type == 'matchup':
            args.append((arg_tid, arg_gid))

        for arg in args:
            result = self.fromCacheOrExtract(arg, types)
            if result is not None:
                row_dict[arg] = result

        # generate response according to request type
        data = self.generateResponse(row_dict, request_type, color)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)
