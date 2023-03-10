#pragma once
#include <iostream>
#include <vector>
#include <algorithm>

#include "jsoncpp/json/json.h"

#include "eigen3/Eigen/Dense"

#include <fstream>
#include <sstream>
#include <unordered_map>
#include <cmath>
#include <queue>

#include "ros/ros.h"
#include <std_msgs/Float32MultiArray.h>
#include <nav_msgs/Path.h>
#include <visualization_msgs/Marker.h>
#include "amrl_msgs/Localization2DMsg.h"


using namespace std;
using namespace Eigen;

struct TimedPCL{
    vector<Vector2f> data; 
    double time;
    TimedPCL(vector<Vector2f> d, double t) : data(d), time(t) {}
    TimedPCL(const TimedPCL &other) : data(other.data), time(other.time) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

struct TimedOdom{
    Vector3f data; // x,y,angle
    double time;
    TimedOdom(Vector3f d, double t) : data(d), time(t) {}
    TimedOdom(const TimedOdom &other) : data(other.data), time(other.time) {}

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
};

const int dx[8] = {-1, 1, 0, 0, -1, 1, -1, 1};
const int dy[8] = {0, 0, -1, 1, -1, -1, 1, 1};

struct Node {
    int x, y;
    float f_score;

    bool operator<(const Node& n) const {
        return f_score > n.f_score;
    }
};

static float getHeuristic(int x1, int y1, int x2, int y2) {
    return max(abs(x1 - x2), abs(y1 - y2)) + 0.001 * sqrt(pow(x1-x2,2) + pow(y1-y2,2));
}

static vector<pair<int, int>> aStar(vector<vector<float>>& costMap, int startX, int startY, int endX, int endY) {
    int m = costMap.size(), n = costMap[0].size(); // should be 256, 256
    if(m != 256 || n != 256){
        cout << "size is not 256" << endl;
        throw exception();
    }
    vector<pair<int, int>> path;
    priority_queue<Node> open_list;
    unordered_map<int, unordered_map<int, float>> gScore;
    unordered_map<int, unordered_map<int, bool>> visited;
    unordered_map<int, unordered_map<int, int>> backtrack;

    open_list.push({startX, startY, 0 + getHeuristic(startX, startY, endX, endY)});

    // initialize gScore
    for(int i=0;i<n;i++){
        for(int j=0;j<m;j++){
            gScore[i][j] = (m+1)*(n+1);
        }
    }
    
    costMap[endY][endX] = 0; // goal should not have a cost

    gScore[startY][startX] = 0.0;
    bool arrived = false;
    int count = 0;

    while (!open_list.empty()) {
        count ++;
        // if(count > 150) break;
        Node curr = open_list.top();
        open_list.pop();

        if (visited[curr.y][curr.x]) continue;
        visited[curr.y][curr.x] = true;
        
        for (int i = 0; i < 8; i++) {
            int nx = curr.x + dx[i], ny = curr.y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                float ds  = sqrt(pow(dx[i], 2) + pow(dy[i],2)); // this astar is not correct, will adjust later
                float tentativeGScore = gScore[curr.y][curr.x] + costMap[ny][nx] + ds;
                if (!visited[ny][nx] && tentativeGScore < gScore[ny][nx]) {
                    gScore[ny][nx] = tentativeGScore;
                    open_list.push({nx, ny, tentativeGScore + getHeuristic(nx, ny, endX, endY)});
                    backtrack[ny][nx] = curr.x + curr.y * m;
                }
                if(nx == endX && ny == endY){
                    // should not break, there might be a faster route?
                    arrived = true;
                    break;
                }
            }
        }

        if (arrived) {
            // backtrack
            int x = endX, y = endY;
            int count = 0;
            int max_count = 2000;
            while ((x != startX || y != startY) && (count < max_count)) {
                // this could stuck in a loop: not sure why, prob bug in astar
                // cout << "x,y: " << x << " " << y << endl;
                path.emplace_back(x, y);
                int idx = backtrack[y][x];
                x = idx % m;
                y = (idx - x) / m;
                count++;
            }
            if(count >= max_count){
                cout << "need to log error: max count exceeded, goal not reachable" << endl;
                path.clear();
                path.emplace_back(startX, startY);
                break;
            }
            path.emplace_back(startX, startY);
            reverse(path.begin(), path.end());
            break;
        }
    }

    return path;
}

static float HausdorffDistance(const vector<pair<float, float>>& v1, const vector<pair<float, float>>& v2) {
  int m = v1.size(), n = v2.size();
  vector<float> dists1(m, 300), dists2(n, 300);
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float dist = sqrt(pow(v1[i].first - v2[j].first, 2) + pow(v1[i].second - v2[j].second, 2));
      dists1[i] = min(dists1[i], dist);
      dists2[j] = min(dists2[j], dist);
    }
  }
  return max(*max_element(dists1.begin(), dists1.end()), *max_element(dists2.begin(), dists2.end()));
}

class DataStore{

    public: 
        DataStore(){};
        void init(ros::NodeHandle nh){
            sub_ = nh.subscribe("/bc_data_store/cost_map", 10, &DataStore::post_processing_costmap, this);
            pub_ = nh.advertise<std_msgs::Float32MultiArray>("/bc_data_store/input_img", 1);
            path_pub_ = nh.advertise<nav_msgs::Path>("/predicted_path", 1);
            adjusted_local_goal_pub_ = nh.advertise<visualization_msgs::Marker>("adjusted_local_goal", 1);
            profiler_stop_pub_ = nh.advertise<std_msgs::Float32MultiArray>("/profiler/stop", 1);
            sub_global_goal_ = nh.subscribe("/move_base_simple/goal_amrl", 10, &DataStore::get_global_goal, this);
            negative_sample_path_pub_ = nh.advertise<std_msgs::Float32MultiArray>("/bc_data_store/path", 1);
        }
        
        vector<TimedPCL> past_point_clouds_;
        vector<TimedOdom> past_odoms_; 
        vector<Eigen::MatrixXf> lidar_scans_;
        vector<float> input_img_vector_;
        vector<float> map_design_;

        float time_interval_ = 0.5;
        float max_time_diff_ = 0.25;
        int img_size_ = 256;
        float resolution_ = 0.078125;
        float range_ = 10.0;
        float dx_ = 128;
        float dy_ = 128;

        bool save_img_ = false;
        int my_count_ = 0;
        double initial_time_;
        bool use_profiler_ = true;

        vector<pair<float, float>> path_;
        Eigen::Vector3f robot_pos_;

        Vector2f global_goal_;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;

        bool Run(){
            auto success = create_odom_lidar_pair();
            if(!success){
                cout << "create odom lidar pair unsuccessful" << endl;
                return false;
            }
            construct_input();
            publish_input();
            return collision_checking();
        }

        bool collision_checking(){
            return false;
            auto curr_odom = past_odoms_.back().data;
            auto curr_observation = lidar_scans_.back();

            // convert odom to map
            auto mat_map = build_transform_matrix({0,0,0}, curr_odom);
            Eigen::Vector3f observed_state_map(0, 0, 1);
            Eigen::Vector3f rotated_state_map = mat_map * observed_state_map;
            auto loc_x = rotated_state_map[0];
            auto loc_y = rotated_state_map[1];

            double min_dist = 100;
            int curr_idx = 0;
            int cnt = 0;
            for(auto p : path_){
                auto dist = pow(p.first - loc_x,2) + pow(p.second - loc_y,2);
                if(min_dist > dist){
                    min_dist = dist;
                    curr_idx = cnt;
                }
                cnt += 1;
            }
            
            for(int i = 5; i < 30; i ++){
                int lookahead_num = curr_idx + i;
                if(path_.size() < lookahead_num + 1){
                    return false;
                }

                auto mat = build_transform_matrix(curr_odom, {0,0,0});

                float x = path_[lookahead_num].first;
                float y = path_[lookahead_num].second;
                Eigen::Vector3f observed_state(x, y, 1);
                Eigen::Vector3f rotated_state = mat * observed_state;

                float x_r = rotated_state[0];
                float y_r = rotated_state[1];
                int ix = (dx_ + int(x_r / resolution_));
                int iy = (dy_ - int(y_r / resolution_));

                for(int x = ix-3 ; x < ix+3; x ++){
                    for(int y = iy-3; y < iy+3; y++){
                        if(ix < 0 || ix >= 256 || iy < 0 || iy > 256){
                            continue;
                        }
                        if(curr_observation(y,x) != 0){
                            cout << curr_observation(y,x) << " " << x << " " << y << endl;
                            return true;
                        }
                    }
                }
            }
            return false;
        }

        void publish_input(){
            std_msgs::Float32MultiArray msg;
            msg.data = input_img_vector_;
            pub_.publish(msg);
        }

        void post_processing_costmap(std_msgs::Float32MultiArray msg){
            std::vector<std::vector<float>> cost_map(256, std::vector<float>(256));
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    cost_map[i][j] = msg.data[i * 256 + j];
                }
            }
            float odom_theta = msg.data[msg.data.size()-1];
            float odom_y = msg.data[msg.data.size()-2];
            float odom_x = msg.data[msg.data.size()-3];
            int iy = msg.data[msg.data.size()-4];
            int ix = msg.data[msg.data.size()-5];

            auto path = calculate_path(cost_map, ix, iy);
            if(path.size() < 10){
                return;
            }

            vector<pair<float, float>> tmp_path;

            // convert to map frame
            auto mat = build_transform_matrix({0,0,0}, {odom_x, odom_y, odom_theta});

            std_msgs::Float32MultiArray negative_path_msg;
            nav_msgs::Path path_msg;
            path_msg.header.frame_id = "odom";
            path_msg.header.stamp = ros::Time::now();
            for(auto p : path){
                negative_path_msg.data.push_back(p.first);
                negative_path_msg.data.push_back(p.second);
                float x = ((float)(p.first - dx_)) * resolution_;
                float y = ((float)(dy_ - p.second)) * resolution_;

                Eigen::Vector3f observed_state(x, y, 1);
                Eigen::Vector3f rotated_state = mat * observed_state;
                float x_r = rotated_state[0];
                float y_r = rotated_state[1];
                tmp_path.push_back({x_r,y_r});

                geometry_msgs::PoseStamped pose;
                pose.header.frame_id = "odom";
                pose.header.stamp = ros::Time::now();
                pose.pose.position.x = x_r;
                pose.pose.position.y = y_r;
                path_msg.poses.push_back(pose);
            }

            auto curr_goal = get_bc_target_dir(path_);
            auto next_goal = get_bc_target_dir(tmp_path);
            if(abs(curr_goal - next_goal) > M_PI){
                cout << "goal dir: " << curr_goal << " " << next_goal << endl;
                return;
            } 
            path_.clear();
            path_ = tmp_path;

            path_pub_.publish(path_msg);
            negative_sample_path_pub_.publish(negative_path_msg);
        }
        
        void store_point_cloud(vector<Vector2f> point_cloud, double time){
            past_point_clouds_.push_back({point_cloud, time});

            if(past_point_clouds_.size() > 50){
                past_point_clouds_.erase(past_point_clouds_.begin());
            }
            
        }
    
        void store_odom(Vector2f odom, float angle, double time){
            Vector3f loc;
            loc << odom[0], odom[1], angle;
            past_odoms_.push_back({loc, time});

            if(past_odoms_.size() > 50){
                past_odoms_.erase(past_odoms_.begin());
            }
        }

        bool create_odom_lidar_pair(){
            // deep copy in reverse order
            vector<TimedOdom> odoms(past_odoms_.rbegin(), past_odoms_.rend());
            vector<TimedPCL> point_clouds(past_point_clouds_.rbegin(), past_point_clouds_.rend());

            if(odoms.size() < 50 || point_clouds.size() < 50 ){
                cout << "not all data are populated yet" << endl;
                return false;
            }
            
            auto curr_odom_time = odoms[0].time;
            auto curr_laser_time = point_clouds[0].time;
            if(abs(curr_odom_time - curr_laser_time) > 0.1){
                cout << "the time difference is too big!!!!!!!!!!!!" << endl;
                return false;
            }
            
            int counter = 0;
            
            vector<TimedPCL> selected_pcl;
            for(auto pcl : point_clouds){
                auto dt = curr_laser_time - pcl.time;
                if(abs(dt - time_interval_ * counter) < max_time_diff_){
                    counter++;
                    selected_pcl.push_back(pcl);
                }
                if(counter == 5){
                    break;
                }
            }
            
            counter = 0;
            vector<TimedOdom> selected_odom;
            for(auto odom : odoms){
                auto dt = curr_odom_time - odom.time;
                if(abs(dt - time_interval_ * counter) < max_time_diff_){
                    counter++;
                    selected_odom.push_back(odom);
                }
                if(counter == 5){
                    break;
                }
            }

            if(selected_pcl.size() != 5 || selected_odom.size() != 5){
                cout << "size is not correct for past odom/pcl " << selected_pcl.size() << " " << selected_odom.size() << endl;
                return false;
            }

            Eigen::Matrix3f rotate_mat;
            lidar_scans_.clear();

            // add image in timestamp seq
            Eigen::MatrixXf kernel = Eigen::MatrixXf::Ones(5,5);
            for(int i = selected_odom.size() - 1; i >= 0; i--){
                rotate_mat = build_transform_matrix(selected_odom[0].data, selected_odom[i].data);
                auto tmp_img = get_bev_lidar_img_rotate(selected_pcl[i].data, rotate_mat);
                
                // apply kernel:
                Eigen::MatrixXf padded_mat = Eigen::MatrixXf::Zero(256+4, 256+4);
                padded_mat.block(2, 2, 256, 256) = tmp_img;

                Eigen::MatrixXf result = Eigen::MatrixXf::Zero(256, 256);
                for (int ik = 2; ik < padded_mat.rows() - 2; ++ik) {
                    for (int j = 2; j < padded_mat.cols() - 2; ++j) {
                        result(ik - 2, j - 2) = (padded_mat.block(ik - 2, j - 2, 5, 5) * kernel.transpose()).sum() > 0 ? 1 : 0;
                    }
                }
                
                lidar_scans_.push_back(result);
            }
            robot_pos_ = selected_odom[0].data;

            if(save_img_){
                stringstream ss;
                for(int i = 0; i < lidar_scans_.size(); i++){
                    ss << i;
                    string fn = "matrix" + ss.str() + ".json";
                    ss.clear();
                    save(lidar_scans_[i], fn);
                }
            }


            return true;
        }

        Eigen::MatrixXf get_current_observation(){
            Eigen::MatrixXf kernel = Eigen::MatrixXf::Ones(11,11);
            auto curr_odom = past_odoms_.back();
            auto curr_pcl = past_point_clouds_.back();
            auto rotate_mat = build_transform_matrix(curr_odom.data, curr_odom.data);
            auto tmp_img = get_bev_lidar_img_rotate(curr_pcl.data, rotate_mat);
            
            // apply kernel:
            Eigen::MatrixXf padded_mat = Eigen::MatrixXf::Zero(256+10, 256+10);
            padded_mat.block(5, 5, 256, 256) = tmp_img;

            Eigen::MatrixXf result = Eigen::MatrixXf::Zero(256, 256);
            for (int ik = 5; ik < padded_mat.rows() - 5; ++ik) {
                for (int j = 5; j < padded_mat.cols() - 5; ++j) {
                    result(ik - 5, j - 5) = (padded_mat.block(ik - 5, j - 5, 11, 11) * kernel.transpose()).sum() > 0 ? 1 : 0;
                }
            }
                
            return result;
        }

        bool check_collision_img(Eigen::MatrixXf img, int x, int y){
            int radius = 10;
            int x_lower = x - radius > 0 ? x - 5 : 0;
            int x_upper = x + radius < 256 ? x + radius : 255;
            int y_lower = y - radius > 0 ? y - radius : 0;
            int y_upper = y + radius < 256 ? y + radius : 255;

            for(int i = x_lower; i <= x_upper; i++){
                for(int j = y_lower; j <= y_upper; j++){
                    if(img(j,i) != 0){
                        return true;
                    }
                }
            }
            return false;
        }

        void construct_input(){
            // get global goal
            auto curr_odom = past_odoms_.back().data;
            auto x = curr_odom[0];
            auto y = curr_odom[1];
            auto angle = curr_odom[2];
            auto theta = atan2(global_goal_[1] - y, global_goal_[0] - x);
            auto dist = sqrt(pow(global_goal_[1] - y, 2) + pow(global_goal_[0] - x, 2));
            auto curr_angle = theta - angle;
            
            // create goal map (away from the obstacle)
            Eigen::MatrixXf goal_map = Eigen::MatrixXf::Zero(img_size_, img_size_);
            auto curr_obs = get_current_observation();

            float x_r, y_r, ix, iy;
            

            if(dist > 6){
                bool set_goal_flag = false;
                for(float r = 8; r >= 5; r -= 0.5){
                    x_r = cos(curr_angle) * r;
                    y_r = sin(curr_angle) * r;
                    ix = (float)(dx_ + int(x_r / resolution_));
                    iy = (float)(dy_ - int(y_r / resolution_));
                    if(check_collision_img(curr_obs, ix, iy)){
                        continue;
                    }
                    goal_map(iy, ix) = 1;
                    set_goal_flag = true;
                    break;
                }
                if(!set_goal_flag){
                    cout << "cannot find a valid carrot!!!" << endl;
                }
            }else{
                x_r = cos(curr_angle) * dist;
                y_r = sin(curr_angle) * dist;
                ix = (float)(dx_ + int(x_r / resolution_));
                iy = (float)(dy_ - int(y_r / resolution_));
                goal_map(iy, ix) = 1;
            }

            // create input tensor
            input_img_vector_.clear();
            for (auto scan : lidar_scans_) {
                for (int j = 0; j < scan.rows(); j++) {
                    for (int k = 0; k < scan.cols(); k++) {
                        input_img_vector_.push_back(scan(j, k));
                    }
                }
            }

            for (int j = 0; j < goal_map.rows(); j++) {
                for (int k = 0; k < goal_map.cols(); k++) {
                    input_img_vector_.push_back(goal_map(j, k));
                }
            }

            // put goal here
            input_img_vector_.push_back(ix);
            input_img_vector_.push_back(iy);

            // input pos odom_x, odom_y, odom_theta
            input_img_vector_.push_back(robot_pos_[0]);
            input_img_vector_.push_back(robot_pos_[1]);
            input_img_vector_.push_back(robot_pos_[2]);
        }

        std::vector<std::pair<int, int>> calculate_path(vector<vector<float>> costmap, int ix, int iy){
            std::vector<std::pair<int, int>> path = aStar(costmap, 128, 128, ix, iy);
            
            // create astar path map
            if(save_img_){
                Eigen::MatrixXf astar_map_cost = Eigen::MatrixXf::Zero(img_size_, img_size_);
                Eigen::MatrixXf astar_map_design = Eigen::MatrixXf::Zero(img_size_, img_size_);
                for (int i = 0; i < 256; i++) {
                    for (int j = 0; j < 256; j++) {
                        astar_map_design(i,j) = map_design_[i * 256 + j];
                        astar_map_cost(i,j) = costmap[i][j];
                    }
                }

                for(auto p : path){
                    auto x = p.first;
                    auto y = p.second;
                    astar_map_cost(y,x) = 100;
                    astar_map_design(y,x) = 1;
                }
                stringstream ss;
                my_count_ ++;
                ss << my_count_;
                string fn = "astar" + ss.str() + ".json";
                string fn_design = "astar_design" + ss.str() + ".json";
                save(astar_map_cost, fn);
                save(astar_map_design, fn_design);
            }
            return path;
        }

        float get_bc_target_dir(vector<pair<float,float>> path){
            float dir = 0;

            auto curr_odom = past_odoms_.back().data;
            // convert odom to map
            auto mat_map = build_transform_matrix({0,0,0}, curr_odom);
            Eigen::Vector3f observed_state_map(0, 0, 1);
            Eigen::Vector3f rotated_state_map = mat_map * observed_state_map;
            auto loc_x = rotated_state_map[0];
            auto loc_y = rotated_state_map[1];

            double min_dist = 100;
            int curr_idx = 0;
            int cnt = 0;
            for(auto p : path){
                auto dist = pow(p.first - loc_x,2) + pow(p.second - loc_y,2);
                if(min_dist > dist){
                    min_dist = dist;
                    curr_idx = cnt;
                }
                cnt += 1;
            }

            int lookahead_num = curr_idx + 40;
            if(path.size() < lookahead_num + 1){
                lookahead_num = 40;
                if(path.size() < lookahead_num + 1){
                    return dir;
                }
            }

            auto mat = build_transform_matrix(curr_odom, {0,0,0});

            float x = path[lookahead_num].first;
            float y = path[lookahead_num].second;
            Eigen::Vector3f observed_state(x, y, 1);
            Eigen::Vector3f rotated_state = mat * observed_state;

            float x_r = rotated_state[0];
            float y_r = rotated_state[1];

            return atan2(y_r, x_r);
        }

        Eigen::Vector2f get_bc_target(){
            Eigen::Vector2f adjusted_goal(10,10);

            auto curr_odom = past_odoms_.back().data;
            // convert odom to map
            auto mat_map = build_transform_matrix({0,0,0}, curr_odom);
            Eigen::Vector3f observed_state_map(0, 0, 1);
            Eigen::Vector3f rotated_state_map = mat_map * observed_state_map;
            auto loc_x = rotated_state_map[0];
            auto loc_y = rotated_state_map[1];

            double min_dist = 100;
            int curr_idx = 0;
            int cnt = 0;
            for(auto p : path_){
                auto dist = pow(p.first - loc_x,2) + pow(p.second - loc_y,2);
                if(min_dist > dist){
                    min_dist = dist;
                    curr_idx = cnt;
                }
                cnt += 1;
            }

            int lookahead_num = curr_idx + 25;
            if(path_.size() < lookahead_num + 1){
                lookahead_num = 25;
                if(path_.size() < lookahead_num + 1){
                    return adjusted_goal;
                }
            }

            auto mat = build_transform_matrix(curr_odom, {0,0,0});

            float x = path_[lookahead_num].first;
            float y = path_[lookahead_num].second;
            Eigen::Vector3f observed_state(x, y, 1);
            Eigen::Vector3f rotated_state = mat * observed_state;

            float x_r = rotated_state[0];
            float y_r = rotated_state[1];

            visualization_msgs::Marker marker;
            marker.header.frame_id = "/base_footprint";
            marker.header.stamp = ros::Time::now();

            marker.type = marker.SPHERE;


            marker.pose.position.x = x_r;
            marker.pose.position.y = y_r;

            marker.scale.x = 0.1;
            marker.scale.y = 0.1;
            marker.scale.z = 0.1;

            marker.color.r = 1.0f;
            marker.color.g = 1.0f;
            marker.color.b = 0.0f;
            marker.color.a = 1.0;

            adjusted_local_goal_pub_.publish(marker);

            adjusted_goal[0] = x_r;
            adjusted_goal[1] = y_r;
            return adjusted_goal;
        } 

        void save(Eigen::MatrixXf img, string name){
             Json::Value root;
            for (int i = 0; i < img.rows(); i++) {
                Json::Value row;
                for (int j = 0; j < img.cols(); j++) {
                    row[j] = img(i, j);
                }
                root[i] = row;
            }

            // Save the JsonCpp value to a file
            std::ofstream file(name);
            file << Json::StyledWriter().write(root);
            file.close();
        }
        
        Eigen::Matrix3f build_transform_matrix(Eigen::Vector3f original_frame, Eigen::Vector3f current_frame){
            auto goal_pt = transform_frame(original_frame,current_frame);
            
            float x_t, y_t;
            x_t = goal_pt[0];
            y_t = goal_pt[1];

            float yaw_o, yaw_c, theta;

            yaw_o = original_frame[2];
            yaw_c = current_frame[2];
            theta = yaw_c - yaw_o;

            Eigen::Matrix3f mat;
            mat << cos(theta), -sin(theta), x_t,
                   sin(theta),  cos(theta), y_t,
                   0,           0,          1;
            
            return mat;
        }

        Eigen::Vector2f transform_frame(Eigen::Vector3f original_frame, Eigen::Vector3f current_frame) {
            // need to test this one
            float x_o, y_o, yaw_o;
            x_o = original_frame[0];
            y_o = original_frame[1];
            yaw_o = original_frame[2];

            float x_c, y_c;
            x_c = current_frame[0];
            y_c = current_frame[1];

            Eigen::Matrix2f mat;
            mat << cos(yaw_o), sin(yaw_o),
                    -sin(yaw_o), cos(yaw_o);

            Eigen::Vector2f coord(x_c - x_o, y_c - y_o);
            Eigen::Vector2f goal_pt = mat * coord;

            return goal_pt;
        }

        Eigen::MatrixXf get_bev_lidar_img_rotate(vector<Vector2f> lidar_points, Eigen::Matrix3f mat) {
            Eigen::MatrixXf img = Eigen::MatrixXf::Zero(img_size_, img_size_);
            for (const auto &lidar_point : lidar_points) {
                float x = lidar_point[0];
                float y = lidar_point[1];


                Eigen::Vector3f observed_state(x, y, 1);
                Eigen::Vector3f rotated_state = mat * observed_state;
                float x_r = rotated_state[0];
                float y_r = rotated_state[1];
                int ix = (dx_ + int(x_r / resolution_));
                int iy = (dy_ - int(y_r / resolution_));

                if (ix >= img_size_ || iy >= img_size_ || ix < 0 || iy < 0) {
                    // out of the frame
                    continue;
                }

                img(iy, ix) = 1;
            }

            return img;
        }

        void get_global_goal(const amrl_msgs::Localization2DMsg& msg){
            const Vector2f loc(msg.pose.x, msg.pose.y);
            global_goal_ = loc;
        }
        void update_vel(){
        }
        void get_vel(Vector2f& vel_cmd, float& ang_vel_cmd){
        }
    private:
        ros::Subscriber sub_;
        ros::Publisher pub_;
        ros::Publisher path_pub_;
        ros::Publisher adjusted_local_goal_pub_;
        ros::Publisher profiler_stop_pub_;
        ros::Publisher negative_sample_path_pub_;
        ros::Subscriber sub_global_goal_;

        pair<float,float> convert_from_goal_idx_to_coord(pair<int,int> pt){
            float x = ((float)(pt.first - dx_)) * resolution_;
            float y = ((float)(dy_ - pt.second)) * resolution_;
            return pair<float, float> (x,y);
        }
};