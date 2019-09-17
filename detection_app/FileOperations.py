import os


def handleUploadedFile(file):
    with open('detection_app/input/' + str(file), 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
        return destination


def validateFileExtension(value):
    ext = os.path.splitext(value)[1]
    valid_extensions = ['.jpg', '.png']
    if not ext in valid_extensions:
        raise Exception(u'File not supported!')


def deleteFile(fileName):
    if os.path.exists('detection_app/input/' + fileName):
        os.remove('detection_app/input/' + fileName)
