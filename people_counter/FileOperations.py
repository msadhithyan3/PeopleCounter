import os


def handleUploadedFile(file):
    with open('people_counter/input/' + str(file), 'wb+') as destination:
        for chunk in file.chunks():
            destination.write(chunk)
        return destination


def validateFileExtension(value):
    ext = os.path.splitext(value)[1]
    print ext
    valid_extensions = ['.jpg', '.png']
    if not ext in valid_extensions:
        raise Exception(u'File not supported!')


def deleteFile(fileName):
    if os.path.exists('people_counter/input/' + fileName):
        os.remove('people_counter/input/' + fileName)
