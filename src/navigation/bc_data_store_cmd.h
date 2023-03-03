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
                float tentativeGScore = gScore[curr.y][curr.x] + costMap[ny][nx] ;
                // sqrt(pow(dx[i], 2) + pow(dy[i],2)) this astar is not correct, will adjust later
                if (!visited[ny][nx] && tentativeGScore < gScore[ny][nx]) {
                    gScore[ny][nx] = tentativeGScore;
                    open_list.push({nx, ny, tentativeGScore + getHeuristic(nx, ny, endX, endY)});
                    backtrack[ny][nx] = curr.x + curr.y * m;
                }
                if(nx == endX && ny == endY){
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
            for(int i =0; i < 63; i++){
                cmd_executed_.push_back(0);
                if(i % 3 ==0){
                    cmd_to_execute_.push_back(1);   
                }
                else{
                    cmd_to_execute_.push_back(0);   
                }
            }
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


        vector<float> cmd_executed_;
        vector<float> cmd_to_execute_;
        int cmd_counter = 0;

        EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
        bool Run(Eigen::Vector2f goal){
            auto success = create_odom_lidar_pair();
            if(!success){
                cout << "create odom lidar pair unsuccessful" << endl;
                return false;
            }
            construct_input(goal);
            publish_input();
            return collision_checking();
        }

        bool collision_checking(){
            // get collisionb checking wrt angle
            auto curr_observation = lidar_scans_.back();
            
            for(int x = 128; x < 138; x ++){
                for(int y = 118; y < 138; y++){
                    if(curr_observation(y,x) != 0){
                        cout << curr_observation(y,x) << " " << x << " " << y << endl;
                        return true;
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

        void get_vel(Vector2f& vel_cmd, float& ang_vel_cmd){
            if(cmd_to_execute_.size() < 3){
                cout << "about to crash" << endl;
            }
            float x = cmd_to_execute_[0] > 1.6? 1.6 : cmd_to_execute_[0];
            x = x < 0 ? 0 : x;
            float y = cmd_to_execute_[1];
            float theta = cmd_to_execute_[2];
            vel_cmd = {x, 0};
            ang_vel_cmd = theta;
            cmd_to_execute_.erase(cmd_to_execute_.begin(), cmd_to_execute_.begin() + 3);
            cmd_executed_.erase(cmd_executed_.begin(), cmd_executed_.begin() + 3);
            cmd_executed_.push_back(x);
            cmd_executed_.push_back(0);
            cmd_executed_.push_back(theta);
        }

        void update_vel(){
            cmd_executed_.clear();
            cmd_to_execute_.clear();
            for(int i =0; i < 63; i++){
                cmd_executed_.push_back(0);
                cmd_to_execute_.push_back(0);
            }
        }

        void post_processing_costmap(std_msgs::Float32MultiArray msg){
            cmd_to_execute_.clear();
            for(auto d : msg.data){
                cmd_to_execute_.push_back(d);
            }
        }
        
        void store_point_cloud(vector<Vector2f> point_cloud, double time){
            past_point_clouds_.push_back({point_cloud, time});

            if(past_point_clouds_.size() > 50){
                past_point_clouds_.erase(past_point_clouds_.begin());
            }
            
        }
    
        void store_odom(Vector2f odom, float angle, double time){
            // if(past_odoms_.size() == 0){
            //     initial_time_ = time;
            // }
            // if(time - initial_time_ > 200){
            //     // let program run 200s
            //     std_msgs::Float32MultiArray msg;
            //     msg.data = {1};
            //     profiler_stop_pub_.publish(msg);
            //     exit(0);
            // }
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
                Eigen::MatrixXf padded_mat = Eigen::MatrixXf::Zero(260, 260);
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

        void construct_input(Eigen::Vector2f goal){
            // create map design
            Eigen::MatrixXf mat = lidar_scans_.back();
            
            std::vector<std::vector<float>> map_design(mat.rows(), std::vector<float>(mat.cols()));
            for (int i = 0; i < mat.rows(); ++i) {
                for (int j = 0; j < mat.cols(); ++j) {
                    map_design[i][j] = mat(i, j) * img_size_ * img_size_;
                }
            }

            // create goal map 
            Eigen::MatrixXf goal_map = Eigen::MatrixXf::Zero(img_size_, img_size_);
            float x_r, y_r;

            x_r = goal[0];
            y_r = goal[1];

            int ix = (dx_ + int(x_r / resolution_));
            int iy = (dy_ - int(y_r / resolution_));
            goal_map(iy, ix) = 1;
            std::vector<std::pair<int, int>> path = aStar(map_design, 128, 128, ix, iy);

            // create astar path map
            Eigen::MatrixXf astar_map = Eigen::MatrixXf::Zero(img_size_, img_size_);
            for(auto p : path){
                auto x = p.first;
                auto y = p.second;
                astar_map(y,x) = 1;
            }

            // create map design
            map_design_.clear();
            auto tmp_map = lidar_scans_[lidar_scans_.size() - 1];
            for (int j = 0; j < tmp_map.rows(); j++) {
                for (int k = 0; k < tmp_map.cols(); k++) {
                    map_design_.push_back(tmp_map(j, k));
                }
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

            for (int j = 0; j < astar_map.rows(); j++) {
                for (int k = 0; k < astar_map.cols(); k++) {
                    input_img_vector_.push_back(astar_map(j, k));
                }
            }
 
            for (int j = 0; j < goal_map.rows(); j++) {
                for (int k = 0; k < goal_map.cols(); k++) {
                    input_img_vector_.push_back(goal_map(j, k));
                }
            }

            // give actions
            input_img_vector_.insert(input_img_vector_.end(), cmd_executed_.begin(), cmd_executed_.end());
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

        Eigen::Vector2f get_bc_target(){
            Eigen::Vector2f adjusted_goal(10,10);
            int lookahead_num = 25;
            if(path_.size() < lookahead_num + 1){
                return adjusted_goal;
            }

            auto curr_odom = past_odoms_.back().data;
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
    private:
        ros::Subscriber sub_;
        ros::Publisher pub_;
        ros::Publisher path_pub_;
        ros::Publisher adjusted_local_goal_pub_;
        ros::Publisher profiler_stop_pub_;

        pair<float,float> convert_from_goal_idx_to_coord(pair<int,int> pt){
            float x = ((float)(pt.first - dx_)) * resolution_;
            float y = ((float)(dy_ - pt.second)) * resolution_;
            return pair<float, float> (x,y);
        }
};