import numpy as np
import tensorflow as tf
import cv2 as cv

# Read the graph.
CLASSES = ["person"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

with tf.gfile.FastGFile('inference_graph/frozen_inference_graph.pb', 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Session() as sess:
    # Restore session
    sess.graph.as_default()
    tf.import_graph_def(graph_def, name='')

    # Read and preprocess an image.
    img = cv.imread('input/person1.jpg')
    rows = img.shape[0]
    cols = img.shape[1]
    inp = cv.resize(img, (300, 300))
    inp = inp[:, :, [2, 1, 0]]  # BGR2RGB

    # Run the model
    out = sess.run([sess.graph.get_tensor_by_name('num_detections:0'),
                    sess.graph.get_tensor_by_name('detection_scores:0'),
                    sess.graph.get_tensor_by_name('detection_boxes:0'),
                    sess.graph.get_tensor_by_name('detection_classes:0')],
                   feed_dict={'image_tensor:0': inp.reshape(1, inp.shape[0], inp.shape[1], 3)})

    # Visualize detected bounding boxes.
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
            cv.putText(img, label, (startX, startY), cv.FONT_HERSHEY_SIMPLEX, 0.5, COLORS[classId], 2)
            cv.rectangle(img, (startX, startY), (endX, endY), COLORS[classId], thickness=1)

cv.imshow('TensorFlow MobileNet-SSD', img)
cv.waitKey()