from datetime import datetime, timedelta
import json
from collections import defaultdict

from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response

from statsETL.bball.apiExtraction import colorExtractor, TeamAPIExtractor, PlayerAPIExtractor, PLAYER_STAT_TYPE, TEAM_STAT_TYPE, BASIC_STAT_TYPE_KEYS, MATCHUP_STAT_TYPE_KEYS
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
                    data_row.update(v['row'])
                    types_left.discard(t)
        else:
            print "NO CACHED RESPONSE FOR %s" % (arg,)
        
        # extract if necessary
        if len(types_left) > 0:
            pid, tid, gid = arg
            vector = getPlayerVector(pid, tid, gid, recalculate=False)
            row = PlayerAPIExtractor().extractVectors(vector, list(types_left), cache=True)
            data_row.update(row)

        # remove row if empty windowed stats
        relevant_keys = [_ for _ in data_row.keys() if 'windowed' in _ or 'exponential' in _]
        if len(relevant_keys) > 0 and all([data_row[_] is None or _ in data_row.keys()]):
            data_row = None

        return data_row

    def generateResponse(self, row_dict, request_type):
        # get headers
        if request_type == 'basic':
            headers = [{'key':_[0],'name':_[1]} for _ in BASIC_STAT_TYPE_KEYS]
        elif request_type == 'matchup':
            headers = [{'key':_[0],'name':_[1]} for _ in MATCHUP_STAT_TYPE_KEYS]
        else:
            raise Exception("Invalid request type")
        # filter rows
        header_keys = [_['key'] for _ in headers]
        if request_type == 'basic':
            # create position row lists
            rows = {'G': [], 'F': [], 'C': []}
            for arg, row in row_dict.iteritems():
                filtered_row = {_:None for _ in header_keys}
                for k in row.keys():
                    if '_' not in k:
                        nonprefix_key = k
                    else:
                        nonprefix_key = '_'.join(k.split('_')[1:])
                    if nonprefix_key in filtered_row:
                        filtered_row[nonprefix_key] = row[k]
                pos = filtered_row['positions']['v'] # get values for position
                if pos is None:
                    continue
                if 'G' in pos:
                    rows['G'].append(filtered_row)
                if 'F' in pos:
                    rows['F'].append(filtered_row)
                if 'C' in pos:
                    rows['C'].append(filtered_row)
            data = {'headers': headers,
                    'rows': rows}
        elif request_type == 'matchup':
            whole_row = row_dict.values()[0]
            # create rows out of each data type
            by_stattype = {}
            for k in whole_row.keys():
                prefix = k.split('_')[0]
                if prefix not in by_stattype:
                    by_stattype[prefix] = {_:None for _ in header_keys}
                nonprefix_key = '_'.join(k.split('_')[1:])
                if nonprefix_key in header_keys:
                    by_stattype[prefix][nonprefix_key] = whole_row[k]
            all_rows = by_stattype.values()
            rows = {'ALL': all_rows}
            data = {'headers': headers,
                    'rows': rows}
        return data

    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            request_type = params['type']
            if request_type == 'matchup':
                arg_pid = params['pid']
                arg_gid = params['gid']
                arg_tid = params['tid']
            elif request_type == 'basic':
                from_date = datetime.strptime(params['from'],'%Y-%m-%d')
                to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
            else:
                raise Exception("Invalid Type specified")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)

        color = colorExtractor('player', date=to_date)
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
            data_row = self.fromCacheOrExtract(arg, types)
            if data_row is not None:
                # fill colors
                for k,v in data_row.iteritems():
                    data_row[k] = {'v':v, 'c': color.extractColor(k, v)}
                row_dict[arg] = data_row

        # generate response according to request type
        data = self.generateResponse(row_dict, request_type)
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
                    data_row.update(v['row'])
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

    def generateResponse(self, row_dict, request_type):
        # get headers
        if request_type == 'basic':
            headers = [{'key':_[0],'name':_[1]} for _ in BASIC_STAT_TYPE_KEYS]
        elif request_type == 'matchup':
            headers = [{'key':_[0],'name':_[1]} for _ in MATCHUP_STAT_TYPE_KEYS]
        else:
            raise Exception("Invalid request type")
        # filter rows
        header_keys = [_['key'] for _ in headers]
        if request_type == 'basic':
            # create position row lists
            all_rows = []
            for arg, row in row_dict.iteritems():
                filtered_row = {_:None for _ in header_keys}
                for k in row.keys():
                    if '_' not in k:
                        nonprefix_key = k
                    else:
                        nonprefix_key = '_'.join(k.split('_')[1:])
                    if nonprefix_key in filtered_row:
                        filtered_row[nonprefix_key] = row[k]
                pos = filtered_row['positions']['v'] # get values for position
                if pos is None:
                    continue
                all_rows.append(filtered_row)
            rows = {'ALL': all_rows}
            data = {'headers': headers,
                    'rows': rows}
        elif request_type == 'matchup':
            whole_row = row_dict.values()[0]
            # create rows out of each data type
            by_stattype = {}
            for k in whole_row.keys():
                prefix = k.split('_')[0]
                if prefix not in by_stattype:
                    by_stattype[prefix] = {_:None for _ in header_keys}
                nonprefix_key = '_'.join(k.split('_')[1:])
                if nonprefix_key in header_keys:
                    by_stattype[prefix][nonprefix_key] = whole_row[k]
            all_rows = by_stattype.values()
            rows = {'ALL': all_rows}
            data = {'headers': headers,
                    'rows': rows}
        return data


    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            request_type = params['type']
            if request_type == 'matchup':
                arg_gid = params['gid']
                arg_tid = params['tid']
            elif request_type == 'basic':
                from_date = datetime.strptime(params['from'],'%Y-%m-%d')
                to_date = datetime.strptime(params['to'], '%Y-%m-%d') + timedelta(days=1)
            else:
                raise Exception("Invalid Type specified")
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid format: %s' % e}, status=status.HTTP_400_BAD_REQUEST)

        color = colorExtractor('team', date=to_date)
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
            data_row = self.fromCacheOrExtract(arg, types)
            if data_row is not None:
                # fill colors
                for k,v in data_row.iteritems():
                    data_row[k] = {'v':v, 'c': color.extractColor(k, v)}
                row_dict[arg] = data_row

        # generate response according to request type
        data = self.generateResponse(row_dict, request_type)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)
