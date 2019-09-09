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


def handle_uploaded_file(file):
    with open('people_counter/input/' + str(file), 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
        return destination


class GetPeopleCount(GenericAPIView):
    def post(self, request, *args, **kwargs):
        try:
            handle_uploaded_file(request.FILES['file'])
            peopleCount, outputImageUrl = counter.getPeopleCount(str(request.FILES['file']))
            message = "People Count successfully Fetched"
            status_code = status.HTTP_200_OK
        except Exception:
            traceback.print_exc()
            peopleCount = 0
            outputImageUrl = None
            message = "Failed to Fetch People Count"
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        response = {
            "data": {
                "message": message,
                "outputVideoUrl": outputImageUrl,
                "peopleCount": peopleCount,
            }
        }
        return Response(response, status=status_code)
