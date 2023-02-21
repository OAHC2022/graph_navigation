#!/usr/bin/env python

import rospy
import subprocess
import signal
from std_msgs.msg import Float32MultiArray
import os

rospy.init_node('controller_node', anonymous=True)
pub = rospy.Publisher("/profiler/stop", Float32MultiArray, queue_size=1)

DIR = "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/"

exp_num = 15
for i in range(30):
    child = subprocess.Popen(["roslaunch","graph_navigation","social_nav.launch", "exp_num:={}".format(exp_num)])
    # child.wait() #You can use this line to block the parent process untill the child process finished.
    print("parent process")
    print(child.poll())

    rospy.loginfo('The PID of child: %d', child.pid)
    print ("The PID of child:", child.pid)

    msg = Float32MultiArray()
    rospy.sleep(500)
    pub.publish(msg)

    child.send_signal(signal.SIGINT)
    rospy.sleep(25)

    os.rename(DIR + "log.log", DIR + "result/merge/exp_baseline_merge_run_{}.log".format(i))


# exp_num = 18
# for i in range(30):
#     child = subprocess.Popen(["roslaunch","graph_navigation","social_nav.launch", "exp_num:={}".format(exp_num)])
#     # child.wait() #You can use this line to block the parent process untill the child process finished.
#     print("parent process")
#     print(child.poll())

#     rospy.loginfo('The PID of child: %d', child.pid)
#     print ("The PID of child:", child.pid)

#     msg = Float32MultiArray()
#     rospy.sleep(500)
#     pub.publish(msg)

#     child.send_signal(signal.SIGINT)
#     rospy.sleep(25)

#     os.rename(DIR + "log.log", DIR + "result/merge/exp_{}_merge_run_{}.log".format(exp_num, i))


# exp_num = 17
# for i in range(30):
#     child = subprocess.Popen(["roslaunch","graph_navigation","social_nav.launch", "exp_num:={}".format(exp_num)])
#     # child.wait() #You can use this line to block the parent process untill the child process finished.
#     print("parent process")
#     print(child.poll())

#     rospy.loginfo('The PID of child: %d', child.pid)
#     print ("The PID of child:", child.pid)

#     msg = Float32MultiArray()
#     rospy.sleep(500)
#     pub.publish(msg)

#     child.send_signal(signal.SIGINT)
#     rospy.sleep(25)

#     os.rename(DIR + "log.log", DIR + "result/merge/exp_{}_merge_run_{}.log".format(exp_num, i))