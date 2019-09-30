import traceback

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2
import tensorflow as tf
from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject


def getPeopleCount():
    try:
        CLASSES = ["person"]

        prototxt = "detection_app/mobilenet_ssd/MobileNetSSD_deploy.prototxt"
        model = "detection_app/mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
        output = "detection_app/output/videos/example_01.avi"
        input = None
        defaultConfidence = 0.4
        with tf.gfile.FastGFile('detection_app/inference_graph/frozen_inference_graph.pb', 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        # load our serialized model from disk
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)

        # if a video path was not supplied, grab a reference to the webcam
        if input is None:
            print("[INFO] starting video stream...")
            vs = VideoStream(src='rtsp://admin:V6YN7j4kfR#@!@192.168.3.118').start()
            time.sleep(2.0)

        # otherwise, grab a reference to the video file
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(input)

        # initialize the video writer (we'll instantiate later if need be)
        writer = None

        # initialize the frame dimensions (we'll set them as soon as we read
        # the first frame from the video)
        W = None
        H = None

        # instantiate our centroid tracker, then initialize a list to store
        # each of our dlib correlation trackers, followed by a dictionary to
        # map each unique object ID to a TrackableObject
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}

        # initialize the total number of frames processed thus far, along
        # with the total number of objects that have moved either up or down
        totalFrames = 0
        peopleCount = 0
        # start the frames per second throughput estimator
        fps = FPS().start()

        # loop over frames from the video stream
        while True:
            # grab the next frame and handle if we are reading from either
            # VideoCapture or VideoStream
            frame = vs.read()
            frame = frame[1] if input is not None else frame

            # if we are viewing a video and we did not grab a frame then we
            # have reached the end of the video
            if input is not None and frame is None:
                break

            # resize the frame to have a maximum width of 500 pixels (the
            # less data we have, the faster we can process it), then convert
            # the frame from BGR to RGB for dlib
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # if the frame dimensions are empty, set them
            if W is None or H is None:
                (H, W) = frame.shape[:2]

            # if we are supposed to be writing a video to disk, initialize
            # the writer
            if output is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output, fourcc, 15, (W, H), True)

            # initialize the current status along with our list of bounding
            # box rectangles returned by either (1) our object detector or
            # (2) the correlation trackers
            currentStatus = "Waiting"
            rects = []

            # check to see if we should run a more computationally expensive
            # object detection method to aid our tracker
            if True:
                # set the status and initialize our new set of object trackers
                currentStatus = "Detecting"
                trackers = []

                with tf.Session() as sess:
                    # Restore session
                    sess.graph.as_default()
                    tf.import_graph_def(graph_def, name='')

                    # Read and preprocess an image.
                    rows = frame.shape[0]
                    cols = frame.shape[1]
                    inp = cv2.resize(frame, (300, 300))
                    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

                    # Run the model
                    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                                    sess.graph.get_tensor_by_name('detection_scores:0'),
                                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                                    sess.graph.get_tensor_by_name('detection_classes:0')],
                                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})
                    num_detections = int(out[0][0])
                    for i in range(num_detections):
                        classId = int(out[3][0][i])
                        classId -= 1
                        score = float(out[1][0][i])
                        bbox = [float(v) for v in out[2][0][i]]
                        if score > 0.4:
                            startX = int(bbox[1] * cols)
                            startY = int(bbox[0] * rows)
                            endX = int(bbox[3] * cols)
                            endY = int(bbox[2] * rows)
                            label = "{}: {:.2f}%".format(CLASSES[classId], score * 100)
                            # cv2.putText(frame, label, (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classId], 2)
                            # cv2.rectangle(frame, (startX, startY), (endX, endY), COLORS[classId], thickness=1)
                            rects.append((startX, startY, endX, endY))
                            tracker = dlib.correlation_tracker()
                            rect = dlib.rectangle(startX, startY, endX, endY)
                            tracker.start_track(rgb, rect)

                            # add the tracker to our list of trackers so we can
                            # utilize it during skip frames
                            trackers.append(tracker)

            # otherwise, we should utilize our object *trackers* rather than
            # object *detectors* to obtain a higher frame processing throughput
            # cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            # use the centroid tracker to associate the (1) old object
            # centroids with (2) the newly computed object centroids
            objects = ct.update(rects)

            # loop over the tracked objects
            for (objectID, centroid) in objects.items():
                # check to see if a trackable object exists for the current
                # object ID
                to = trackableObjects.get(objectID, None)

                # if there is no existing trackable object, create one
                if to is None:
                    to = TrackableObject(objectID, centroid)

                # otherwise, there is a trackable object so we can utilize it
                else:
                    # check to see if the object has been counted or not
                    if not to.counted:
                        # count the object
                        peopleCount += 1
                        to.counted = True
                # store the trackable object in our dictionary
                trackableObjects[objectID] = to

                # draw both the ID of the object and the centroid of the
                # object on the output frame
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                # cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
                cv2.rectangle(frame, ((centroid[0] - 30, centroid[1] - 40)), ((centroid[0] + 30, centroid[1] + 40)),
                              (0, 255, 0), 1)

            # construct a tuple of information we will be displaying on the
            # frame
            info = [
                ("PeopleCount", peopleCount)
            ]

            # loop over the info tuples and draw them on our frame
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # check to see if we should write the frame to disk
            if writer is not None:
                writer.write(frame)

            # show the output frame
            cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF

            # if the `q` key was pressed, break from the loop
            if key == ord("q"):
                break

            # increment the total number of frames processed thus far and
            # then update the FPS counter
            totalFrames += 1
            fps.update()

        # stop the timer and display FPS information
        fps.stop()
        outputVideoUrl = "localhost:8000/static/videos/example_01.avi"
        print("PeopleCount", peopleCount)
        print("outputVideoUrl", outputVideoUrl)
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))

        # check to see if we need to release the video writer pointer
        if writer is not None:
            writer.release()

        # if we are not using a video file, stop the camera video stream
        if input is None:
            vs.stop()

        # otherwise, release the video file pointer
        else:
            vs.release()

        # close any open windows
        cv2.destroyAllWindows()
        return (peopleCount, outputVideoUrl)
    except Exception as ex:
        traceback.print_exc()
        raise ex