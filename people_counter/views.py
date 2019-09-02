# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
import traceback

from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from people_counter import counter


class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)


class GetPeopleCount(GenericAPIView):
    def get(self, request, *args, **kwargs):
        try:
            peopleCount,outputVideoUrl = counter.getPeopleCount()
            message = "People Count successfully Fetched"
            status_code = status.HTTP_200_OK
        except Exception:
            traceback.print_exc()
            peopleCount = 0
            outputVideoUrl=None
            message = "Failed to Fetch People Count"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        response = {
            "data": {
                "message": message,
                "peopleCount": peopleCount,
                "outputVideoUrl":outputVideoUrl
            }
        }
        return Response(response, status=status_code)
