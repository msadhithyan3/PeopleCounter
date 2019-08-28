import traceback

from imutils.video import VideoStream
from imutils.video import FPS
import numpy as np
import imutils
import time
import dlib
import cv2

from pyimagesearch.centroidtracker import CentroidTracker
from pyimagesearch.trackableobject import TrackableObject


def getPeopleCount():
    try:
        CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
                   "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
                   "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
                   "sofa", "train", "tvmonitor"]
        prototxt = "people_counter/mobilenet_ssd/MobileNetSSD_deploy.prototxt"
        model = "people_counter/mobilenet_ssd/MobileNetSSD_deploy.caffemodel"
        input = "people_counter/videos/honda.mp4"
        output = "people_counter/output/videos/honda.avi"
        skip_frames = 30
        defaultConfidence = 0.4
        print("[INFO] loading model...")
        net = cv2.dnn.readNetFromCaffe(prototxt, model)
        if input is None:
            print("[INFO] starting video stream...")
            vs = VideoStream(src=0).start()
            time.sleep(2.0)
        else:
            print("[INFO] opening video file...")
            vs = cv2.VideoCapture(input)
        writer = None
        W = None
        H = None
        ct = CentroidTracker(maxDisappeared=40, maxDistance=50)
        trackers = []
        trackableObjects = {}
        totalFrames = 0
        totalDown = 0
        totalUp = 0
        fps = FPS().start()
        while True:
            frame = vs.read()
            frame = frame[1] if input is not None else frame
            if input is not None and frame is None:
                break
            frame = imutils.resize(frame, width=500)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            if W is None or H is None:
                (H, W) = frame.shape[:2]
            if output is not None and writer is None:
                fourcc = cv2.VideoWriter_fourcc(*"MJPG")
                writer = cv2.VideoWriter(output, fourcc, 30, (W, H), True)
            currentStatus = "Waiting"
            rects = []
            if totalFrames % skip_frames == 0:
                currentStatus = "Detecting"
                trackers = []
                blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
                net.setInput(blob)
                detections = net.forward()
                for i in np.arange(0, detections.shape[2]):
                    confidence = detections[0, 0, i, 2]
                    if confidence > defaultConfidence:
                        idx = int(detections[0, 0, i, 1])
                        if CLASSES[idx] != "person":
                            continue
                        box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                        (startX, startY, endX, endY) = box.astype("int")
                        tracker = dlib.correlation_tracker()
                        rect = dlib.rectangle(startX, startY, endX, endY)
                        tracker.start_track(rgb, rect)
                        trackers.append(tracker)
            else:
                for tracker in trackers:
                    currentStatus = "Tracking"
                    tracker.update(rgb)
                    pos = tracker.get_position()
                    startX = int(pos.left())
                    startY = int(pos.top())
                    endX = int(pos.right())
                    endY = int(pos.bottom())
                    rects.append((startX, startY, endX, endY))
            cv2.line(frame, (0, H // 2), (W, H // 2), (0, 255, 255), 2)
            objects = ct.update(rects)
            for (objectID, centroid) in objects.items():
                to = trackableObjects.get(objectID, None)
                if to is None:
                    to = TrackableObject(objectID, centroid)
                else:
                    y = [c[1] for c in to.centroids]
                    direction = centroid[1] - np.mean(y)
                    to.centroids.append(centroid)
                    if not to.counted:
                        if direction < 0 and centroid[1] < H // 2:
                            totalUp += 1
                            to.counted = True
                        elif direction > 0 and centroid[1] > H // 2:
                            totalDown += 1
                            to.counted = True
                trackableObjects[objectID] = to
                text = "ID {}".format(objectID)
                cv2.putText(frame, text, (centroid[0] - 10, centroid[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            (0, 255, 0), 2)
                cv2.circle(frame, (centroid[0], centroid[1]), 4, (0, 255, 0), -1)
            info = [
                ("Up", totalUp),
                ("Down", totalDown),
                ("Status", currentStatus),
            ]
            for (i, (k, v)) in enumerate(info):
                text = "{}: {}".format(k, v)
                cv2.putText(frame, text, (10, H - ((i * 20) + 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            if writer is not None:
                writer.write(frame)
            # cv2.imshow("Frame", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            totalFrames += 1
            fps.update()
        fps.stop()
        print("totalUp", totalUp)
        print("totalDown", totalDown)
        print("[INFO] elapsed time: {:.2f}".format(fps.elapsed()))
        print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
        if writer is not None:
            writer.release()
        if input is None:
            vs.stop()
        else:
            vs.release()
        cv2.destroyAllWindows()
        return totalDown + totalUp
    except Exception as ex:
        traceback.print_exc()
        raise ex
