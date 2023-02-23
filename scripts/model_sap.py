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

from bc3 import BC3
from processing_utils import *
import time 

DEVICE = "cuda:0"
class BC3_Ported(nn.Module):
    def __init__(self, exp_num):
        super(BC3_Ported, self).__init__()
        self.exp_num = exp_num
        self.unet_encoder = BC3(exp_num=exp_num).unet_encoder
        self.traj_embed =  BC3(exp_num=exp_num).traj_embed
        self.head =  BC3(exp_num=exp_num).head

    def forward(self, lidar_scans, goal, past_traj):
        if self.exp_num in [1,2]:
            _, z = self.unet_encoder(lidar_scans)

            traj_goal = torch.cat([past_traj, goal], dim=1)
            
            traj_goal = self.traj_embed(traj_goal)

            z1 = torch.cat([z, traj_goal], dim=1)

            traj = self.head(z1)
            return traj
        elif self.exp_num in [3]:
            _, z = self.unet_encoder(lidar_scans)
            traj_goal = self.traj_embed(goal)
            z1 = torch.cat([z, traj_goal], dim=1)
            traj = self.head(z1)
            return traj



def get_checkpoint_name(exp_num, epoch_num):
    MODEL_DIR = "/robodata/zichaohu/training/BC3/{}/*".format(exp_num)
    files = glob.glob(MODEL_DIR)
    for f in files:
        if "pred-epoch={}".format(epoch_num) in f:
            fn = f.split('/')
            return fn[-1]

def load_model(exp_num, checkpoint):
    COUNT = exp_num
    MODEL_DIR = "/robodata/zichaohu/training/BC3/{}/".format(COUNT)
    model = BC3_Ported(exp_num=COUNT)
    
    dict = torch.load(MODEL_DIR + checkpoint, map_location=DEVICE)['state_dict']
    new_dict = {}
    for k,v in dict.items():
        new_dict[k] = v
    model.load_state_dict(new_dict)
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
    past_traj = - data[-45:-5]
    input_img = data[:-45]

    angle = torch.tensor(angle, dtype=torch.float).to(DEVICE)
    angle = angle.view(-1,1)

    past_traj = torch.tensor(past_traj).to(DEVICE)
    past_traj = past_traj.view(1,40)

    input_img = torch.tensor(input_img).to(DEVICE)
    input_img = input_img.view(1,5,256,256)
    predicted_traj = model(input_img, angle, past_traj)

    predicted_traj = predicted_traj.flatten().detach().cpu().numpy()
    # px = 0
    # py = 0
    # predicted_traj = []
    # for i in range(0, len(traj), 2):
    #     px += traj[i]
    #     py += traj[i+1]
    #     predicted_traj.append(px)
    #     predicted_traj.append(py)
    # predicted_traj = np.array(predicted_traj)


    predicted_traj = np.append(predicted_traj, odom_x)
    predicted_traj = np.append(predicted_traj, odom_y)
    predicted_traj = np.append(predicted_traj, odom_theta)
    predicted_traj = np.append(predicted_traj, odom_time)
    
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
    past_traj = torch.randn(1,40).to(DEVICE)
    angle = torch.randn(1,1).to(DEVICE)
    result = model(input_img, angle, past_traj)
    end = time.time()
    print("preheat first: ", end -start)

    start = time.time()
    input_img = torch.randn(1,5,256,256).to(DEVICE)
    past_traj = torch.randn(1,40).to(DEVICE)
    angle = torch.randn(1,1).to(DEVICE)
    result = model(input_img, angle, past_traj)
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
    check_point = get_checkpoint_name(exp_num,80)
    # check_point = "pred-epoch=73-val_loss=0.03488.ckpt"
    model = load_model(exp_num,check_point)
    preheat_model()
    rospy.init_node('model_node', anonymous=True)
    rospy.Subscriber("/bc_data_store/input_img", Float32MultiArray, callback)
    pub = rospy.Publisher("/bc_data_store/cost_map", Float32MultiArray, queue_size=1)
    rospy.spin()
