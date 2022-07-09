import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
import cv2
import os
import json
import time

interpreter = tf.lite.Interpreter('lite-model_movenet_singlepose_lightning_3.tflite')
interpreter.allocate_tensors()



EDGES = {
    (0, 1): 'm',
    (0, 2): 'c',
    (1, 3): 'm',
    (2, 4): 'c',
    (0, 5): 'm',
    (0, 6): 'c',
    (5, 7): 'm',
    (7, 9): 'm',
    (6, 8): 'c',
    (8, 10): 'c',
    (5, 6): 'y',
    (5, 11): 'm',
    (6, 12): 'c',
    (11, 12): 'y',
    (11, 13): 'm',
    (13, 15): 'm',
    (12, 14): 'c',
    (14, 16): 'c'
}

dict = {"Confidence": [], "Posex": [], "Posey": []}


def draw_keypoints(frame, keypoints, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for kp in shaped:
        ky, kx, kp_conf = kp
        dict["Confidence"].append(kp_conf)
        dict["Posex"].append(kx)
        dict["Posey"].append(ky)
        # print(str(q))
        # data = json.load(q)
        # print (json.dumps(data, sort_keys=True, indent=4))
        # make new JSON file
        # with open('/content/drive/MyDrive/Colab Notebooks/movenet/data.json', 'w') as f:
        #   f.write(str(json.dumps(encode)))
        if kp_conf > confidence_threshold:
            cv2.circle(frame, (int(kx), int(ky)), 4, (0, 255, 0), -1)

        # json_string = json.dumps(dict, default=lambda o: o.__dict__, sort_keys=True, indent=2)
        # return json_string
        return dict



def draw_connections(frame, keypoints, edges, confidence_threshold):
    y, x, c = frame.shape
    shaped = np.squeeze(np.multiply(keypoints, [y, x, 1]))

    for edge, color in edges.items():
        p1, p2 = edge
        y1, x1, c1 = shaped[p1]
        y2, x2, c2 = shaped[p2]

        if (c1 > confidence_threshold) & (c2 > confidence_threshold):
            cv2.line(frame, (int(x1), int(y1)), (int(x2), int(y2)), (0, 0, 255), 2)


import os
os.environ["OPENCV_FFMPEG_CAPTURE_OPTIONS"] = "rtsp_transport;udp"
cap = cv2.VideoCapture(0)
dict2 = {}
while cap.isOpened():
    try:
        ret, frame = cap.read()
        # Reshape image
        img = frame.copy()
        img = tf.image.resize_with_pad(np.expand_dims(img, axis=0), 192, 192)
        input_image = tf.cast(img, dtype=tf.float32)

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Make predictions
        interpreter.set_tensor(input_details[0]['index'], np.array(input_image))
        interpreter.invoke()
        keypoints_with_scores = interpreter.get_tensor(output_details[0]['index'])

        # Rendering
        draw_connections(frame, keypoints_with_scores, EDGES, 0.4)
        draw_keypoints(frame, keypoints_with_scores, 0.4)
        dict2.update(dict)

        cv2.imshow(frame)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    except Exception as e:
        print("Exception happened: {}".format(e))
        continue
    return dict2

with open('keypoints.json') as f:
    json.dump(dict2, f)

pose_net('keypoints.json')



cv2.destroyAllWindows()
cap.release()
