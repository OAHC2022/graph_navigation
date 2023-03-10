#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import torch 
import torch.nn as nn
import numpy as np
import glob 
import json
import sys 

sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/BCSAN3')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/neural-astar/src')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/segmentation_models.pytorch')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/scripts')

from bc_vin import BC3
from processing_utils import *
import time 
import torchvision

DEVICE = "cuda:0"


def get_checkpoint_name(exp_num, epoch_num):
    MODEL_DIR = "/robodata/zichaohu/training/BC3_VIN/{}/*".format(exp_num)
    files = glob.glob(MODEL_DIR)
    for f in files:
        if "pred-epoch={}".format(epoch_num) in f:
            fn = f.split('/')
            return fn[-1]

def load_model(exp_num, checkpoint):
    COUNT = exp_num
    MODEL_DIR = "/robodata/zichaohu/training/BC3_VIN/{}/".format(COUNT)
    class Args:
        l_i = 2
        l_h = 150
        l_q = 10

    config=Args()

    model = BC3(exp_num=COUNT, config=config)
    
    dict = torch.load(MODEL_DIR + checkpoint, map_location=DEVICE)['state_dict']
    model.load_state_dict(dict)
    model.eval()
    return model.to(DEVICE)

def callback(data: Float32MultiArray):
    start_time = time.time()
    global count 
    count += 1
    
    data = np.array(data.data, dtype=np.float32)
    odom_time = data[-1]
    odom_theta = data[-2]
    odom_y = data[-3]
    odom_x = data[-4]
    angle = data[-5] 
    input_img = data[:-5]
    print(angle)

    goal_map = torch.full([1,1,256,256], angle, dtype=torch.float).to(DEVICE)

    input_img = torch.tensor(input_img).to(DEVICE)
    input_img = input_img.view(1,5,256,256)

    gb_kernel = torchvision.transforms.GaussianBlur(5, 1.1)
    input_img = gb_kernel(input_img)

    predicted_traj = model(input_img, goal_map)

    predicted_traj = predicted_traj.detach().cpu().numpy()
    print(predicted_traj)

    predicted_traj = np.append(predicted_traj, odom_x)
    predicted_traj = np.append(predicted_traj, odom_y)
    predicted_traj = np.append(predicted_traj, odom_theta)
    predicted_traj = np.append(predicted_traj, odom_time)
    predicted_traj = np.append(predicted_traj, angle)
    
    traj_msg = Float32MultiArray()
    traj_msg.data = predicted_traj

    # print('time: {}'.format(end - start))
    pub.publish(traj_msg)
    end_time = time.time()
    print(count, " ", end_time - start_time)
    # with open ('/home/zichaohu/catkin_ws/src/tmp_{}.pkl'.format(count), 'wb') as f:
    #     additional_info = additional_info.detach().cpu().numpy()
    #     pickle.dump(additional_info, f)


def preheat_model():
    # first time to run the model is very slow, so preheat the model up
    start = time.time()
    input_img = torch.randn(1,5,256,256).to(DEVICE)
    goal_map = torch.randn(1,1,256,256).to(DEVICE)
    result = model(input_img, goal_map)
    end = time.time()
    print("preheat first: ", end -start)

    start = time.time()
    input_img = torch.randn(1,5,256,256).to(DEVICE)
    goal_map = torch.randn(1,1,256,256).to(DEVICE)
    result = model(input_img, goal_map)
    end = time.time()
    print("preheat second: ", end -start)
    if end - start < 0.05:
        print("preheat complete!!")
    else:
        print("preheat and still slow: ", end - start)

count = 0
if __name__ == '__main__':
    exp_num = 1
    print("Start Model Node for exp: {}".format(exp_num))
    check_point = get_checkpoint_name(exp_num,13)
    print("checkpoint:", check_point)
    # check_point = "pred-epoch=73-val_loss=0.03488.ckpt"
    model = load_model(exp_num,check_point)
    preheat_model()
    rospy.init_node('model_node', anonymous=True)
    rospy.Subscriber("/bc_data_store/input_img", Float32MultiArray, callback)
    pub = rospy.Publisher("/bc_data_store/cost_map", Float32MultiArray, queue_size=1)
    rospy.spin()
