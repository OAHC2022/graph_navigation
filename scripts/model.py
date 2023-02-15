#!/usr/bin/env python
import rospy
from std_msgs.msg import Float32MultiArray
import torch 
import torch.nn as nn
import numpy as np
import glob 
import json
import sys 

sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/BCSAN2')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/neural-astar/src')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/segmentation_models.pytorch')
sys.path.append('/home/zichaohu/catkin_ws/src/SocialNavigation/scripts')

from behavior_cloning2 import BC2
from processing_utils import *
import time 

DEVICE = "cuda:0"
class BC2_Ported(nn.Module):
    def __init__(self, exp_num):
        super(BC2_Ported, self).__init__()
        # exp 15-18
        self.global_planner = BC2(exp_num=exp_num, Tmax=1000).global_planner
        self.local_planner = BC2(exp_num=exp_num, Tmax=1000).local_planner
        self.trans_encoder_layer = BC2(exp_num=exp_num, Tmax=1000).trans_encoder_layer
        self.trans_encoder = BC2(exp_num=exp_num, Tmax=1000).trans_encoder
        self.lp_expand = BC2(exp_num=exp_num, Tmax=1000).lp_expand
        self.scaling = BC2(exp_num=exp_num, Tmax=1000).scaling
        self.softplus =  BC2(exp_num=exp_num, Tmax=1000).softplus
        self.prediction_model = BC2(exp_num=exp_num, Tmax=1000).prediction_model
        self.exp_num = exp_num
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img, map_design):
        if self.exp_num in [17,18]:
            goal_map = input_img[:, -1]
            past_lidar_scans = input_img[:, :5]

            tmp_goal_map = goal_map.unsqueeze(1).unsqueeze(1).repeat_interleave(past_lidar_scans.shape[1], dim=1) 
            tmp_past_lidar_scans = past_lidar_scans.unsqueeze(2)
            lidar_scans_input = torch.cat([tmp_past_lidar_scans, tmp_goal_map], dim=2)
            predicted_lidar_scans = self.prediction_model(lidar_scans_input)
            tmp_predicted_lidar_scans = self.sigmoid(predicted_lidar_scans)[:, -5:].squeeze(2)

            additional_info = input_img[:,-2:]
            input_img = torch.cat([past_lidar_scans, tmp_predicted_lidar_scans, additional_info], dim=1)
        
        raw_guidance_map, z1 = self.global_planner(input_img)
        
        cost_map = torch.clamp(self.softplus(map_design + raw_guidance_map) + 1, 1, 50)
        return cost_map, input_img


def get_checkpoint_name(exp_num, epoch_num):
    MODEL_DIR = "/robodata/zichaohu/training/BC2/{}/*".format(exp_num)
    files = glob.glob(MODEL_DIR)
    for f in files:
        if "pred-epoch={}".format(epoch_num) in f:
            fn = f.split('/')
            return fn[-1]

def load_model(exp_num, checkpoint):
    COUNT = exp_num
    MODEL_DIR = "/robodata/zichaohu/training/BC2/{}/".format(COUNT)
    model = BC2_Ported(exp_num=COUNT)
    
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
    input_img = data[:-3]
    # if count % 20 == 0:
    #     with open('/home/zichaohu/catkin_ws/src/input_vec_{}.json'.format(count), 'w') as f:
    #         json.dump(input_img.tolist(), f)

    input_img = torch.tensor(input_img).to(DEVICE)
    input_img = input_img.view(1,7,256,256)
    map_design = input_img[:, 4:5]
    cost_map, additional_info = model(input_img, map_design)

    cost_map = cost_map.flatten().detach().cpu().numpy()
    goal_map = input_img.detach().cpu().numpy()[0,-1]

    y_list, x_list = np.nonzero(goal_map)
    cost_map = np.append(cost_map, x_list[0])
    cost_map = np.append(cost_map, y_list[0])
    cost_map = np.append(cost_map, odom_x)
    cost_map = np.append(cost_map, odom_y)
    cost_map = np.append(cost_map, odom_theta)
    
    cost_map_msg = Float32MultiArray()
    cost_map_msg.data = cost_map

    # print('time: {}'.format(end - start))
    pub.publish(cost_map_msg)
    end_time = time.time()
    print(count, " ", end_time - start_time)
    # with open ('/home/zichaohu/catkin_ws/src/tmp_{}.pkl'.format(count), 'wb') as f:
    #     additional_info = additional_info.detach().cpu().numpy()
    #     pickle.dump(additional_info, f)


def preheat_model():
    # first time to run the model is very slow, so preheat the model up
    start = time.time()
    input_img = torch.randn(1,7,256,256).to(DEVICE)
    map_design = input_img[:, 4:5]
    result = model(input_img, map_design)
    end = time.time()
    print("preheat first: ", end -start)

    start = time.time()
    input_img = torch.randn(1,7,256,256).to(DEVICE)
    map_design = input_img[:, 4:5]
    result = model(input_img, map_design)
    end = time.time()
    print("preheat second: ", end -start)
    if end - start < 0.05:
        print("preheat complete!!")
    else:
        print("preheat and still slow: ", end - start)

count = 0
if __name__ == '__main__':
    exp_num = 18
    print("Start Model Node for exp: {}".format(exp_num))
    check_point = get_checkpoint_name(exp_num,199)
    model = load_model(exp_num,check_point)
    preheat_model()
    rospy.init_node('model_node', anonymous=True)
    rospy.Subscriber("/bc_data_store/input_img", Float32MultiArray, callback)
    pub = rospy.Publisher("/bc_data_store/cost_map", Float32MultiArray, queue_size=1)
    rospy.spin()
