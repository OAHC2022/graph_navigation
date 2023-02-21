#!/usr/bin/env python
import rospy
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from message_filters import ApproximateTimeSynchronizer, Subscriber
from std_msgs.msg import Float32MultiArray

import sys 
sys.path.append('/home/zichaohu/catkin_ws/src/amrl_msgs/src')
from amrl_msgs.msg import Localization2DMsg
import lupa
import numpy as np
import json

collided = False
goal_count = 0
colision_count = 0

def check_collision(human: PoseStamped, odom: Odometry):
    global collided, colision_count
    human_pose = np.array([human.pose.position.x, human.pose.position.y])
    robot_pose = np.array([odom.pose.pose.position.x, odom.pose.pose.position.y])
    dist = np.linalg.norm(human_pose - robot_pose)
    if dist < 0.5 and not collided:
        colision_count += 1
        print("too close: " , colision_count)
        collided = True 
    elif dist >= 0.5:
        collided = False


def goal_cb(goal):
    global goal_count
    goal_count +=1 
    print("count", goal_count, goal)

def stop_cb(data):
    global goal_count, colision_count
    print('stop profiling')
    DIR = "/robodata/zichaohu/training/BC2/sim_results/"
    with open("log.log", "w") as f:
        result = [goal_count, colision_count]
        json.dump(result, f)
    exit()

if __name__ == '__main__':
    print("Start Profiler")
    lua = lupa.LuaRuntime()

    # Read the Lua file
    with open("config/navigation.lua", "r") as file:
        lua_code = file.read()

    # Execute the Lua code
    lua.execute(lua_code)
    odom_topic = lua.globals().NavigationParameters.odom_topic

    rospy.init_node('profiler_node', anonymous=True)
    human_sub = Subscriber("/profiler/humans", PoseStamped)
    odom_sub = Subscriber(odom_topic, Odometry)
    rospy.Subscriber("/profiler/stop", Float32MultiArray, stop_cb)

    ts = ApproximateTimeSynchronizer(
        [human_sub, odom_sub], 100, 0.1, allow_headerless=True)
    ts.registerCallback(check_collision)

    rospy.Subscriber("/profiler/goal", Localization2DMsg, goal_cb)
    rospy.spin()