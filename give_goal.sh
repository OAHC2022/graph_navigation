#!/bin/bash

rostopic pub -1 /move_base_simple/goal_amrl amrl_msgs/Localization2DMsg "{'pose' : {'x' : 10}}"