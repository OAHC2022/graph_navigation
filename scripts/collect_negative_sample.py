#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import torch 
import torch.nn as nn
import numpy as np
import glob 
import json
import sys 

sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/BCSAN_new')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/neural-astar/src')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/segmentation_models.pytorch')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/scripts')

from bc_new import BCNew
from processing_utils import *
import time 
from astar import solve_single
import torchvision
from message_filters import ApproximateTimeSynchronizer, Subscriber


def callback(path: Float32MultiArray, cost_map: Float32MultiArray, input_img: Float32MultiArray):
    global count
    dir = "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/negative_samples/"
    with open(dir + "run_{}.pkl".format(count), "wb") as f:
        pickle.dump([path.data, cost_map.data, input_img.data], f)
    print("negative:", len(path.data)) 
    count += 1


count = 0
if __name__ == '__main__':    
    rospy.init_node('negative_sample_node', anonymous=True)
    path = Subscriber("/bc_data_store/path", Float32MultiArray)
    cost_map = Subscriber("/bc_data_store/cost_map", Float32MultiArray)
    input_img = Subscriber("/bc_data_store/input_img", Float32MultiArray)
    
    ts = ApproximateTimeSynchronizer(
        [path, cost_map, input_img], 100, 0.1, allow_headerless=True)
    ts.registerCallback(callback)

    rospy.spin()
