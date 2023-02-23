#!/usr/bin/env python

import rospy
import subprocess
import signal
from std_msgs.msg import Float32MultiArray
import os
import time 



DIR = "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/"
import sys
exp_num = sys.argv[1]

child = subprocess.Popen(["roslaunch","graph_navigation","social_nav.launch", "exp_num:={}".format(exp_num)])
# child.wait() #You can use this line to block the parent process untill the child process finished.
print("parent process")
print(child.poll())

rospy.loginfo('The PID of child: %d', child.pid)
print ("The PID of child:", child.pid)

msg = Float32MultiArray()

rospy.sleep(300)

rospy.init_node('controller_node', anonymous=True)
pub = rospy.Publisher("/profiler/stop", Float32MultiArray, queue_size=1)
pub.publish(msg)

rospy.sleep(10)

os.killpg(os.getpgid(child.pid), signal.SIGINT)
print('interrupt')


child.wait()
print('process finished')

exit()
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
    time.sleep(25)
    print("create log")
    os.rename(DIR + "log.log", DIR + "result/cross_cmd/exp_{}_cross_run_{}.log".format(exp_num, i))

exp_num = 18
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
    time.sleep(25)

    os.rename(DIR + "log.log", DIR + "result/cross_cmd/exp_{}_cross_run_{}.log".format(exp_num, i))


exp_num = 17
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
    time.sleep(25)

    os.rename(DIR + "log.log", DIR + "result/cross_cmd/exp_{}_cross_run_{}.log".format(exp_num, i))