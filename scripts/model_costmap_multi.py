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

DEVICE = "cuda:0"
class BCNew_Ported(nn.Module):
    def __init__(self, exp_num):
        super(BCNew_Ported, self).__init__()
        # exp 15-18
        self.global_planner = BCNew(exp_num=exp_num, Tmax=1000).global_planner
        self.scale = BCNew(exp_num=exp_num, Tmax=1000).scale
        self.exp_num = exp_num
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, lidar_scans):
        map_design = lidar_scans[:,-1:]
        guidance_map = self.global_planner(lidar_scans)
        cost_map = torch.clamp(map_design + self.tanh(guidance_map), 0, 1) * self.scale
        return cost_map 

def get_checkpoint_name(exp_num, epoch_num):
    MODEL_DIR = "/robodata/zichaohu/training/BCNew/{}/*".format(exp_num)
    files = glob.glob(MODEL_DIR)
    for f in files:
        if "pred-epoch={}".format(epoch_num) in f:
            fn = f.split('/')
            return fn[-1]

def load_model(exp_num, checkpoint):
    COUNT = exp_num
    MODEL_DIR = "/robodata/zichaohu/training/BCNew/{}/".format(COUNT)
    model = BCNew_Ported(exp_num=COUNT)
    
    dict = torch.load(MODEL_DIR + checkpoint, map_location=DEVICE)['state_dict']
    new_dict = {}
    for k,v in dict.items():
        if "vanilla_astar" in k or "diff_astar" in k:
            continue 
        new_dict[k] = v
    model.load_state_dict(new_dict)
    model.eval()
    return model.to(DEVICE)

def callback(data: Float32MultiArray):
    start_time = time.time()
    global count 
    count += 1
    
    data = np.array(data.data, dtype=np.float32)
    odom_theta = data[-1]
    odom_y = data[-2]
    odom_x = data[-3]
    goal_y = data[-4]
    goal_x = data[-5]
    lidar_scans = data[:-5]


    lidar_scans = torch.tensor(lidar_scans).to(DEVICE)
    lidar_scans = lidar_scans.view(1,6,256,256)

    cost_map = model_4(lidar_scans)

    cost_map = cost_map.flatten().detach().cpu().numpy()

    cost_map = np.append(cost_map, goal_x)
    cost_map = np.append(cost_map, goal_y)
    cost_map = np.append(cost_map, odom_x)
    cost_map = np.append(cost_map, odom_y)
    cost_map = np.append(cost_map, odom_theta)
    
    cost_map_msg = Float32MultiArray()
    cost_map_msg.data = cost_map

    pub.publish(cost_map_msg)
    end_time = time.time()
    with open("/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/sim_vis/sim_model_4_{}.pkl".format(count), "wb") as f:
        pickle.dump([data, cost_map], f)

    cost_map = model_5(lidar_scans)
    cost_map = cost_map.flatten().detach().cpu().numpy()

    cost_map = np.append(cost_map, goal_x)
    cost_map = np.append(cost_map, goal_y)
    cost_map = np.append(cost_map, odom_x)
    cost_map = np.append(cost_map, odom_y)
    cost_map = np.append(cost_map, odom_theta)

    with open("/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/sim_vis/sim_model_5_{}.pkl".format(count), "wb") as f:
        pickle.dump([data, cost_map], f)


def preheat_model(model):
    # first time to run the model is very slow, so preheat the model up
    start = time.time()
    input_img = torch.randn(1,6,256,256).to(DEVICE)
    result = model(input_img)
    end = time.time()
    print("preheat first: ", end -start)

    start = time.time()
    input_img = torch.randn(1,6,256,256).to(DEVICE)
    result = model(input_img)
    end = time.time()
    print("preheat second: ", end -start)
    if end - start < 0.05:
        print("preheat complete!!")
    else:
        print("preheat and still slow: ", end - start)

count = 0
if __name__ == '__main__':
    
    print("Start Model Node for exp: {}".format(4))
    check_point = "pred-epoch=70-val_loss=0.00386.ckpt"
    model_4 = load_model(4,check_point)
    check_point = "pred-epoch=70-val_loss=0.00318.ckpt"
    model_5 = load_model(5,check_point)
    preheat_model(model_4)
    preheat_model(model_5)
    rospy.init_node('model_node', anonymous=True)
    rospy.Subscriber("/bc_data_store/input_img", Float32MultiArray, callback)
    pub = rospy.Publisher("/bc_data_store/cost_map", Float32MultiArray, queue_size=1)
    rospy.spin()
