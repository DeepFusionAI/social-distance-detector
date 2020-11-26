import numpy as np
import imutils
import time
import cv2
import os
import math

from itertools import chain 
from constants import *

# Load the labels
LABELS = open(YOLOV3_LABELS_PATH).read().strip().split('\n')

np.random.seed(42)
COLORS = np.random.randint(0, 255, size=(len(LABELS), 3), dtype='uint8')

print('Loading YOLO from disk...')

# Load the network
neural_net = cv2.dnn.readNetFromDarknet(YOLOV3_CFG_PATH, YOLOV3_WEIGHTS_PATH)

# Get the output layer from YOLO
layer_names = neural_net.getLayerNames()
layer_names = [layer_names[i[0] - 1] for i in neural_net.getUnconnectedOutLayers()]

#Input video is read
vs = cv2.VideoCapture(VIDEO_PATH)
writer = None
(W, H) = (None, None)

#Count total frame numbers in input video
try:
    if(imutils.is_cv2()):
        prop = cv2.cv.CV_CAP_PROP_FRAME_COUNT
    else:
        prop = cv2.CAP_PROP_FRAME_COUNT
    total = int(vs.get(prop))
    print('Total frames detected are: ', total)
except Exception as e:
    print(e)
    total = -1

while True:
    (grabbed, frame) = vs.read()

    if not grabbed:
        break
    
    if W is None or H is None:
        H, W = (frame.shape[0], frame.shape[1])
    
    # Read and convert the image to blob and perform forward pass to get the bounding boxes with their confidence scores
    blob = cv2.dnn.blobFromImage(frame, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    neural_net.setInput(blob)

    start_time = time.time()
    layer_outputs = neural_net.forward(layer_names)
    end_time = time.time()
    
    boxes = []
    confidences = []
    classIDs = []
    lines = []
    box_centers = []

    for output in layer_outputs:
        for detection in output:
            
            scores = detection[5:]
            classID = np.argmax(scores)
            confidence = scores[classID]
            
            if confidence > 0.5 and classID == 0:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype('int')
                
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                
                box_centers = [centerX, centerY]

                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)

    # Perform non maximum suppression for the bounding boxes to filter overlapping and low confident bounding boxes
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.3)
    
    # Draw the filtered bounding boxes with their class to the image and calculate the distance between each box's centroid
    if len(idxs) > 0:
        unsafe = []
        count = 0
        
        for i in idxs.flatten():
            
            (x, y) = (boxes[i][0], boxes[i][1])
            (w, h) = (boxes[i][2], boxes[i][3])
            centeriX = boxes[i][0] + (boxes[i][2] // 2)
            centeriY = boxes[i][1] + (boxes[i][3] // 2)

            color = [int(c) for c in COLORS[classIDs[i]]]
            text = '{}: {:.4f}'.format(LABELS[classIDs[i]], confidences[i])

            idxs_copy = list(idxs.flatten())
            idxs_copy.remove(i)

            for j in np.array(idxs_copy):
                centerjX = boxes[j][0] + (boxes[j][2] // 2)
                centerjY = boxes[j][1] + (boxes[j][3] // 2)

                distance = math.sqrt(math.pow(centerjX - centeriX, 2) + math.pow(centerjY - centeriY, 2))

                if distance <= SAFE_DISTANCE:
                    cv2.line(frame, (boxes[i][0] + (boxes[i][2] // 2), boxes[i][1]  + (boxes[i][3] // 2)), (boxes[j][0] + (boxes[j][2] // 2), boxes[j][1] + (boxes[j][3] // 2)), (0, 0, 255), 2)
                    unsafe.append([centerjX, centerjY])
                    unsafe.append([centeriX, centeriY])

            if centeriX in chain(*unsafe) and centeriY in chain(*unsafe):
                count += 1
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            cv2.putText(frame, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            cv2.rectangle(frame, (50, 50), (450, 90), (0, 0, 0), -1)
            cv2.putText(frame, 'No. of people unsafe: {}'.format(count), (70, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 3)            


    if writer is None:

        fourcc = cv2.VideoWriter_fourcc(*'MJPG')
        writer = cv2.VideoWriter(OUTPUT_PATH, fourcc, 30,(frame.shape[1], frame.shape[0]), True)

        if total > 0:
            elap = (end_time - start_time)
            print('Single frame took {:.4f} seconds'.format(elap))
            print('Estimated total time to finish: {:.4f}'.format(elap * total))

    writer.write(frame)

print('Cleaning up...')
writer.release()
vs.release()                                
