from social_distancing_config import *
from pedestrian_detection import detector
from social_distance_detection import *

import cv2
import time
import numpy as np
import tensorflow as tf
#from imutils.video import FPS
import argparse


#@title Select the TFLite model
tflite_model = "../models/ssd_mobiledet_cpu_coco_fp16.tflite" #@param ["ssd_mobiledet_cpu_coco_dr.tflite", "ssd_mobiledet_cpu_coco_fp16.tflite", "ssd_mobiledet_cpu_coco_int8.tflite"] 
interpreter = tf.lite.Interpreter(model_path=tflite_model)
interpreter.allocate_tensors()
_, HEIGHT, WIDTH, _ = interpreter.get_input_details()[0]['shape']
print(f"Height and width accepted by the model: {HEIGHT, WIDTH}")


def process(video,output_path):
    cap=cv2.VideoCapture(video)
    """ Capture the first frame of video """
    res,image=cap.read()
    if image is None:
        return
    image=cv2.resize(image,(WIDTH,HEIGHT))
    
    """ Get the transformation matrix from the first frame """
    mat=getmap(image)
    fourcc=cv2.VideoWriter_fourcc(*"XVID")
    out=cv2.VideoWriter(output_path,fourcc,20.0,(WIDTH*3,HEIGHT))
    #fps = FPS().start()
    print("Processing...")
    while True:
        res,image=cap.read()
        if image is None:
            break
        
        """ pedestrian detection """
        preprocessed_frame = preprocess_frame(image)
        results = detector(interpreter, preprocessed_frame, threshold=CONFIDENCE)
        preprocessed_frame = np.squeeze(preprocessed_frame) * 255.0
        preprocessed_frame = preprocessed_frame.clip(0, 255)
        preprocessed_frame = preprocessed_frame.squeeze()
        image = np.uint8(preprocessed_frame)
        
        """ calibration """
        warped_centroids=calibration(mat, results)
        
        """ Distance-Violation Determination """
        violate=calc_dist(warped_centroids)
        
        """ Visualise grid """
        grid,warped=visualise_grid(image,mat,warped_centroids,violate)
        
        """ Visualise main frame """
        image=visualise_main(image,results,violate)
        
        """ Creating final output frame """
        output=cv2.hconcat((image,warped))
        output=cv2.hconcat((output,grid))
        out.write(output)
        #fps.update()
    #fps.stop()
    #print("[INFO] elasped time: {:.2f}".format(fps.elapsed()))
    #print("[INFO] approx. FPS: {:.2f}".format(fps.fps()))
    cap.release()
    out.release()
    print("Done!")
   
video_path="../videos/pedestrians.mp4"
output_path="../videos/output.mp4"
process(video_path, output_path)
    
# =============================================================================
# if __name__ == '__main__':
#     parser = argparse.ArgumentParser()
#     parser.add_argument('input', help='input video path')
#     parser.add_argument('output', help='output video path with extension')
#     args = parser.parse_args()
#     process(args.input,args.output)
# =============================================================================
