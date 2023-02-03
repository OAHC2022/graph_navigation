#pragma once

#include "onnxruntime/include/onnxruntime_cxx_api.h"
#include <vector>
#include <iostream>
#include "eigen3/Eigen/Dense"
#include "jsoncpp/json/json.h"
#include <torch/torch.h>

using namespace std;
vector<int64_t> static INPUT_IMG_SHAPE = {1,7,256,256};

class BC2{
    public:
        int img_size_ = 256;
        vector<float> raw_guidance_map_result_;

        BC2(){}

        void predict(vector<float> input_img_vector){
            const char* model_name = "/home/zichaohu/catkin_ws/src/SocialNavigation/third_party/graph_navigation/model_15_epoch_199_gp_only.onnx";
            Ort::Env env(ORT_LOGGING_LEVEL_WARNING, "test");
            Ort::SessionOptions session_options;
            Ort::Session session(env, model_name, session_options);

            // get names
            Ort::AllocatorWithDefaultOptions ort_alloc;

            Ort::AllocatedStringPtr input_names = session.GetInputNameAllocated(0, ort_alloc);
            Ort::AllocatedStringPtr output_names = session.GetOutputNameAllocated(0, ort_alloc);

            vector<const char*> input_node_names = {input_names.get()};    
            vector<const char*> output_node_names = {output_names.get()};

            // get tensor
            auto memory_info = Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

            Ort::Value input_img_tensor = Ort::Value::CreateTensor<float>(memory_info, input_img_vector.data(), input_img_vector.size(), INPUT_IMG_SHAPE.data(), INPUT_IMG_SHAPE.size());
            Ort::Value input_tensor[] = {move(input_img_tensor)};

            auto output_tensors = session.Run(Ort::RunOptions{ nullptr }, input_node_names.data(), input_tensor, 1, output_node_names.data(), 1);

            auto raw_guidance_map_tensor = output_tensors[0].GetTensorMutableData<float>();
            auto raw_guidance_map_size = output_tensors[0].GetTensorTypeAndShapeInfo().GetElementCount();

            for(int i = 0; i < raw_guidance_map_size; i++){
                raw_guidance_map_result_.push_back(raw_guidance_map_tensor[i]);
            }

        }

        std::vector<std::vector<float>> post_processing(vector<float> map_design){
            // torch::Tensor tensor = torch::from_blob(map_design.data(), {map_design.size()});
            //  torch::Tensor tensor = torch::from_blob(map_design.data(), {map_design.size()});
            std::vector<std::vector<float>> cost_map(256, std::vector<float>(256));

            for (int i = 0; i < 256; i++) {
                for (int j = 0; j < 256; j++) {
                auto cost = map_design[i * 256 + j] + raw_guidance_map_result_[i * 256 + j];
                auto tmp_cost = cost > 20 ? cost + 1 : log(1 + exp(cost)) + 1 ;
                cost_map[i][j] = tmp_cost > 50 ? 50 : tmp_cost;
                }
            }

            cout << "post" << endl;
            return cost_map;
        }

};