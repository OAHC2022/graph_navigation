#!/usr/bin/env python

import rospy 
import lupa
from nav_msgs.msg import Odometry
import sys 
sys.path.append('/home/zichaohu/catkin_ws/src/amrl_msgs/src')
from amrl_msgs.msg import Localization2DMsg

import time
import numpy as np

goals = [[0,0,0], [10,0, np.pi]]
curr_idx = 0

delay = 400
delay_count = 400 

def odom_cb(odom: Odometry):
    global curr_idx, goals, delay_count
    x = odom.pose.pose.position.x 
    y = odom.pose.pose.position.y 

    curr_pos = np.array([x,y])
    curr_goal = np.array(goals[curr_idx])
    if delay_count % 10 == 0:
        goal = Localization2DMsg()
        goal.pose.x = curr_goal[0]
        goal.pose.y = curr_goal[1]
        goal.pose.theta = curr_goal[2]
        pub.publish(goal)

    if np.linalg.norm(curr_goal[:2] - curr_pos) < 5e-1:
        delay_count += 1
        if delay_count < delay:
            return 
        curr_idx = 1 - curr_idx
        next_goal = np.array(goals[curr_idx])
        goal = Localization2DMsg()
        goal.pose.x = next_goal[0]
        goal.pose.y = next_goal[1]
        goal.pose.theta = next_goal[2]
        pub.publish(goal)
        profiler_pub.publish(goal)
        delay_count = 0
        # print('goal reached!! next goal: ', goal)
    else:
        delay_count = 0

if __name__ == "__main__":
    print("starting give way point node")
    time.sleep(10)
    print('finish sleeping')
    # Create a Lua runtime
    lua = lupa.LuaRuntime()

    # Read the Lua file
    with open("config/navigation.lua", "r") as file:
        lua_code = file.read()

    # Execute the Lua code
    lua.execute(lua_code)
    odom_topic = lua.globals().NavigationParameters.odom_topic
    
    rospy.init_node('waypoint_node', anonymous=True)
    rospy.Subscriber(odom_topic, Odometry, odom_cb)
    pub = rospy.Publisher("/move_base_simple/goal_amrl", Localization2DMsg, queue_size=10)
    profiler_pub = rospy.Publisher("/profiler/goal", Localization2DMsg, queue_size=10)
    rospy.spin()