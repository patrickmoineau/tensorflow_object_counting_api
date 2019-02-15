#----------------------------------------------
#--- Author         : Ahmet Ozlu
#--- Mail           : ahmetozlu93@gmail.com
#--- Date           : 27th January 2018
#--- Modified       : Patrick Moineau 2.8.2019
#----------------------------------------------

# Imports
import tensorflow as tf
import argparse

# Object detection imports
from utils import backbone
from api import object_counting_api

if tf.__version__ < '1.12.0':
  raise ImportError('Please upgrade your tensorflow installation to v1.12.* or later!')

if __name__ == '__main__':
    # Paramaters 
    parser = argparse.ArgumentParser(description='Detect objects and count them')
    parser.add_argument('-V','--video', type=str, default=None, help="path to video file")
    parser.add_argument('-F','--fps', type=int, default=30, help="change it with your input video fps")
    parser.add_argument('-W','--width', type=int, default=626, help="change it with your input video width")
    parser.add_argument('-H','--height', type=int, default=360, help="change it with your input vide height")
    parser.add_argument('-C','--is_color_recognition_enabled', type=int, default=0, help="set it to 1 for enabling the color prediction for the detected objects")
    parser.add_argument('-R','--roi', type=int, default=386, help="roi line position")
    parser.add_argument('-D','--deviation', type=int, default=1, help="the constant that represents the object counting area")
    parser.add_argument('-O','--output', type=str, default="./{}_out.avi", help="path of output avi")
    args = parser.parse_args()

    if args.video == None:
        input_video = "./input_images_and_videos/pedestrian_survaillance.mp4"

    # By default I use an "SSD with Mobilenet" model here. See the detection model zoo (https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) for a list of other models that can be run out-of-the-box with varying speeds and accuracies.
    #detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_2017_11_17')
    detection_graph, category_index = backbone.set_model('ssd_mobilenet_v1_coco_box')

    object_counting_api.cumulative_object_counting_x_axis(args.video, args.output, detection_graph, category_index, args.is_color_recognition_enabled, args.fps, args.width, args.height, args.roi, args.deviation) # counting all the objects
