from django.shortcuts import render
from rest_framework import permissions, viewsets, status, views
from rest_framework.response import Response
from datetime import datetime
import json
from collections import defaultdict

from statsETL.bball.statsExtraction import getTeamsForDay, getPlayersForDay, getPlayerVector, getTeamVector
from statsETL.util.crawler import daterange


'''
Return precalculated NBA player stat vectors in JSON format
'''

class DailyPlayerOwnVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

class DailyPlayerAgainstPosVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

class DailyPlayerPosTrendVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

class DailyPlayerHomeRoadSplitVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

class DailyPlayerOppSplitVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)

class DailyPlayerMetaVectors(views.APIView):

    def get(self, request):
        return Response({}, status=status.HTTP_200_OK)


'''
Return precalculated NBA team stat vectors in JSON format
'''

class DailyTeamOwnVectors(views.APIView):

    def get(self, request):
        params = self.request.GET
        if 'from' not in params or 'to' not in params:
            return Response({'status': 'Bad request', 'message': 'Missing date parameters'}, status=status.HTTP_400_BAD_REQUEST)

        # generate stats query
        try:
            from_date = datetime.strptime(params['from'],'%Y-%m-%d')
            to_date = datetime.strptime(params['to'], '%Y-%m-%d')
        except Exception as e:
            return Response({'status': 'Bad request','message': 'Invalid date format'}, status=status.HTTP_400_BAD_REQUEST)
        args = []
        for date in daterange(from_date, to_date):
            new_args = getTeamsForDay(date)
            args += new_args

        return Response({'args': json.dumps(args)}, status=status.HTTP_400_BAD_REQUEST)

        # query for stats
        data = defaultdict(dict)
        for arg in args:
            tid, gid = arg
            stats = getTeamVector(tid, gid, recalculate=False)
            # get relevant stats to place in vector
            whole_vector = stats['means'].ix['mean'].to_dict()
            filtered = whole_vector
            data[gid][tid] = filtered

        # generate response
        data = dict(data)
        data_json = json.dumps(data)
        return Response(data_json, status=status.HTTP_200_OK)

class DailyTeamOppVectors(views.APIView):

    def get(self,request):
        return Response({}, status=status.HTTP_200_OK)

class DailyTeamSeasonVectors(views.APIView):

    def get(self,request):
        return Response({}, status=status.HTTP_200_OK)

class DailyTeamMetaVectors(views.APIView):

    def get(self,request):
        return Response({}, status=status.HTTP_200_OK)