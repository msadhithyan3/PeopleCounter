import os


def handleUploadedFile(directory, file):
    with open(directory, 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
        return destination


def validateFileExtension(fileName):
    ext = os.path.splitext(fileName)[1]
    valid_extensions = ['.jpg', '.png']
    if not ext in valid_extensions:
        raise Exception(u'File not supported!')


def deleteFile(filePath):
    if os.path.exists(filePath):
        os.remove(filePath)
