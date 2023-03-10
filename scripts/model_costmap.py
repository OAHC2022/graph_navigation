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
        map_design = lidar_scans[:,-2:-1] #bug
        guidance_map = self.global_planner(lidar_scans)
        cost_map = torch.clamp(map_design + self.tanh(guidance_map), 0, 1) * self.scale #just use the guidance map
        return cost_map 

def get_checkpoint_name(exp_num, epoch_num):
    MODEL_DIR = "/robodata/zichaohu/training/BCNew/{}/*".format(exp_num)
    files = glob.glob(MODEL_DIR)
    for f in files:
        if "pred-epoch={}".format(epoch_num) in f:
            fn = f.split('/')
            return fn[-1]

def load_model(exp_num, model_dir):
    COUNT = exp_num
    model = BCNew_Ported(exp_num=COUNT)
    
    dict = torch.load(model_dir, map_location=DEVICE)['state_dict']
    new_dict = {}
    for k,v in dict.items():
        if "vanilla_astar" in k or "diff_astar" in k:
            continue 
        new_dict[k] = v
    model.load_state_dict(new_dict)
    model.eval()
    return model.to(DEVICE)

def post_processing_cost_map(costmap, goal_x, goal_y):
    costmap[0,0,120:136,128:136] = 0
    goal_x = int(goal_x)
    goal_y = int(goal_y)
    costmap[0,0,goal_y-5:goal_y+5, goal_x-5:goal_x+5] = 0


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

    # apply gaussian blur
    gb_kernel = torchvision.transforms.GaussianBlur(5, 1.1)
    lidar_scans = gb_kernel(lidar_scans)

    cost_map = model(lidar_scans)
    
    # post process costmap
    # post_processing_cost_map(cost_map, goal_x, goal_y)

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
    dir = "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/sim_vis/pkl/"
    create_dir_if_not_exist(dir)
    # with open(dir + "sim_{}.pkl".format(count), "wb") as f:
    #     pickle.dump([data, cost_map], f)


def preheat_model():
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
    exp_num = int(sys.argv[1])
    print("Start Model Node for exp: {}".format(exp_num))
    check_point = get_checkpoint_name(exp_num,60)
    # check_point = "pred-epoch=70-val_loss=0.00386.ckpt"
    # check_point = "pred-epoch=90-val_loss=0.00410.ckpt" # exp 7
    # check_point = "ahg_pred-epoch=99-val_loss=0.00278.ckpt" # ahg exp 4
    MODEL_DIR = "/robodata/zichaohu/training/BCNew/{}/ahg/".format(exp_num)
    # MODEL_DIR = "/robodata/zichaohu/training/BCNew/{}/".format(exp_num)
    check_point = "ahg_pred-epoch=170-val_loss=0.00113.ckpt" # ahg exp 7
    # check_point = "pred-epoch=111-val_loss=0.00442.ckpt" # exp 6
    model = load_model(exp_num,MODEL_DIR+check_point)
    preheat_model()
    rospy.init_node('model_node', anonymous=True)
    rospy.Subscriber("/bc_data_store/input_img", Float32MultiArray, callback)
    pub = rospy.Publisher("/bc_data_store/cost_map", Float32MultiArray, queue_size=1)
    rospy.spin()
