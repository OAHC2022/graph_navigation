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
    for(int i=0;i<m;i++){
        for(int j=0;j<n;j++){
            gScore[i][j] = (m+1)*(n+1);
        }
    }
    gScore[startX][startY] = 0.0;
    bool arrived = false;
    int count = 0;

    while (!open_list.empty()) {
        count ++;
        // if(count > 150) break;
        Node curr = open_list.top();
        open_list.pop();

        if (visited[curr.x][curr.y]) continue;
        visited[curr.x][curr.y] = true;
        
        for (int i = 0; i < 8; i++) {
            int nx = curr.x + dx[i], ny = curr.y + dy[i];
            if (nx >= 0 && nx < m && ny >= 0 && ny < n) {
                float tentativeGScore = gScore[curr.x][curr.y] + costMap[ny][nx] ;
                // sqrt(pow(dx[i], 2) + pow(dy[i],2)) this astar is not correct, will adjust later
                if (!visited[nx][ny] && tentativeGScore < gScore[nx][ny]) {
                    gScore[nx][ny] = tentativeGScore;
                    open_list.push({nx, ny, tentativeGScore + getHeuristic(nx, ny, endX, endY)});
                    backtrack[nx][ny] = curr.x * n + curr.y;
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
            while (x != startX || y != startY) {
                path.emplace_back(x, y);
                int idx = backtrack[x][y];
                y = idx % n;
                x = (idx - y) / n;
            }
            path.emplace_back(startX, startY);
            reverse(path.begin(), path.end());
            break;
        }
    }

    return path;
}

class DataStore{
    public: 
        DataStore(){};
        
        vector<TimedPCL> past_point_clouds_;
        vector<TimedOdom> past_odoms_; 
        vector<Eigen::MatrixXf> lidar_scans_;
        vector<float> input_img_vector_;
        vector<float> map_design_;

        float time_interval_ = 0.5;
        float max_time_diff_ = 0.15;
        int img_size_ = 256;
        float resolution_ = 0.078125;
        float range_ = 10.0;
        float dx_ = 128;
        float dy_ = 128;

        bool save_img_ = true;

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
            for(int i = selected_odom.size() - 1; i >= 0; i--){
                rotate_mat = build_transform_matrix(selected_odom[0].data, selected_odom[i].data);
                auto tmp_img = get_bev_lidar_img_rotate(selected_pcl[i].data, rotate_mat);
                lidar_scans_.push_back(tmp_img);
            }

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
                    map_design_.push_back(mat(j, k));
                }
            }

            // create input tensor
            input_img_vector_.clear();
            for (auto scan : lidar_scans_) {
                for (int j = 0; j < scan.rows(); j++) {
                    for (int k = 0; k < scan.cols(); k++) {
                        input_img_vector_.push_back(mat(j, k));
                    }
                }
            }

            for (int j = 0; j < astar_map.rows(); j++) {
                for (int k = 0; k < astar_map.cols(); k++) {
                    input_img_vector_.push_back(mat(j, k));
                }
            }

            for (int j = 0; j < goal_map.rows(); j++) {
                for (int k = 0; k < goal_map.cols(); k++) {
                    input_img_vector_.push_back(mat(j, k));
                }
            }
        }

        void calculate_path(vector<vector<float>> costmap, Eigen::Vector2f goal){
            // create goal  
            Eigen::MatrixXf goal_map = Eigen::MatrixXf::Zero(img_size_, img_size_);
            float x_r, y_r;

            x_r = goal[0];
            y_r = goal[1];

            int ix = (dx_ + int(x_r / resolution_));
            int iy = (dy_ - int(y_r / resolution_));

            std::vector<std::pair<int, int>> path = aStar(costmap, 128, 128, ix, iy);
            // create astar path map
            Eigen::MatrixXf astar_map = Eigen::MatrixXf::Zero(img_size_, img_size_);
            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                    astar_map(i,j) = map_design_[i * 256 + j];
                }
            }

            for(auto p : path){
                auto x = p.first;
                auto y = p.second;
                astar_map(y,x) = 1;
            }
            save(astar_map, "astar.json");
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

};