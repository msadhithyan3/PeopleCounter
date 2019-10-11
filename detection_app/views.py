# -*- coding: utf-8 -*-
from __future__ import unicode_literals

# Create your views here.
import os
import traceback

from django.shortcuts import render
from django.views.generic import TemplateView
from rest_framework import status
from rest_framework.response import Response
from rest_framework.generics import GenericAPIView
from detection_app import counter
import FileOperations as fileOperations


class HomePageView(TemplateView):
    def get(self, request, **kwargs):
        return render(request, 'index.html', context=None)


class GetPeopleCount(GenericAPIView):
    def post(self, request, *args, **kwargs):
        try:
            CWD_PATH = os.getcwd()

            APP_DIRECTORY = 'detection_app'
            INPUT_DIRECTORY = 'input'
            fileName = str(request.FILES['file'])
            fileUploadedDirectory = os.path.join(CWD_PATH, APP_DIRECTORY, INPUT_DIRECTORY)+'/'+fileName
            fileOperations.validateFileExtension(fileName)
            fileOperations.handleUploadedFile(fileUploadedDirectory,request.FILES['file'])
            peopleCount, outputImageUrl = counter.getPeopleCount(fileName)
            fileOperations.deleteFile(fileUploadedDirectory)
            message = "People Count successfully Fetched"
            status_code = status.HTTP_200_OK
        except Exception as ex:
            traceback.print_exc()
            peopleCount = 0
            outputImageUrl = None
            message = ex.message
            status_code = status.HTTP_500_INTERNAL_SERVER_ERROR
        response = {
            "data": {
                "message": message,
                "outputImageUrl": outputImageUrl,
                "peopleCount": peopleCount,
            }
        }
        return Response(response, status=status_code)
