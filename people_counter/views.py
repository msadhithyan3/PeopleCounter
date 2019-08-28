# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
import traceback

from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from people_counter.counter import Test


class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)


class GetPeopleCount(GenericAPIView):
    def get(self, request, *args, **kwargs):
        try:
            test=Test()
            peopleCount = test.getPeopleCount()
            message = "People Count successfully Fetched"
            status_code = status.HTTP_200_OK
        except Exception:
            traceback.print_exc()
            peopleCount = 0
            message = "Failed to Fetch People Count"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        success = {
            "data": {
                "message": message,
                "peopleCount": peopleCount
            }
        }
        return Response(success, status=status_code)
