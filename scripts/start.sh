#!/bin.bash 

for i in {0..15}
do
    python scripts/start_launch.py 18
    sleep 10
    pkill -9 python
    mv /home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/log.log "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/result/cross_cmd_4_5/exp_18_run_$i.log"
    
    python scripts/start_launch.py 15
    sleep 10
    pkill -9 python
    mv /home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/log.log "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/result/cross_cmd_4_5/exp_15_run_$i.log"

    python scripts/start_launch.py 17
    sleep 10
    pkill -9 python
    mv /home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/log.log "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/result/cross_cmd_4_5/exp_17_run_$i.log"
done

# for i in {0..10}
# do
#     python scripts/start_launch.py 18
#     sleep 10
#     pkill -9 python
#     mv /home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/log.log "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/result/difficult_cmd/exp_18_difficult_run_$i.log"
# done

# for i in {0..10}
# do
#     python scripts/start_launch.py 17
#     sleep 10
#     pkill -9 python
#     mv /home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/log.log "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/result/difficult_cmd/exp_17_difficult_run_$i.log"
# done