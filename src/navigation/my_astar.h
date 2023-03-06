#include <vector>
#include <cmath>
#include <unordered_map>
#include <pqxx/pqxx> // to use the priority queue
#include <iostream>

typedef std::pair<int, double> node_t; // pair of node index and cost

class compare
{
public:
    bool operator()(const node_t &n1, const node_t &n2) const
    {
        return n1.second > n2.second;
    }
};

std::vector<int> get_neighbor_indices(int idx, int H, int W)
{
    std::vector<int> neighbor_indices;
    if (idx % W - 1 >= 0)
        neighbor_indices.push_back(idx - 1);
    if (idx % W + 1 < W)
        neighbor_indices.push_back(idx + 1);
    if (idx / W - 1 >= 0)
        neighbor_indices.push_back(idx - W);
    if (idx / W + 1 < H)
        neighbor_indices.push_back(idx + W);
    if ((idx % W - 1 >= 0) && (idx / W - 1 >= 0))
        neighbor_indices.push_back(idx - W - 1);
    if ((idx % W + 1 < W) && (idx / W - 1 >= 0))
        neighbor_indices.push_back(idx - W + 1);
    if ((idx % W - 1 >= 0) && (idx / W + 1 < H))
        neighbor_indices.push_back(idx + W - 1);
    if ((idx % W + 1 < W) && (idx / W + 1 < H))
        neighbor_indices.push_back(idx + W + 1);

    return neighbor_indices;
}

double compute_chebyshev_distance(int idx, int goal_idx, int W)
{
    int loc[2] = {idx % W, idx / W};
    int goal_loc[2] = {goal_idx % W, goal_idx / W};
    double dxdy[2] = {std::abs(loc[0] - goal_loc[0]), std::abs(loc[1] - goal_loc[1])};
    double h = dxdy[0] + dxdy[1] - std::min(dxdy[0], dxdy[1]);
    double euc = std::sqrt(std::pow(loc[0] - goal_loc[0], 2) + std::pow(loc[1] - goal_loc[1], 2));
    return h + 0.001 * euc;
}

std::vector<std::vector<int>> get_history(std::unordered_map<int, double> &close_list, int H, int W)
{
    std::vector<std::vector<int>> history;
    for (const auto &node : close_list)
    {
        int idx = node.first;
        std::vector<int> coords = {idx % W, idx / W};
        history.push_back(coords);
    }
    return history;
}

std::vector<std::vector<int>> backtrack(std::unordered_map<int, int> &parent_list, int goal_idx, int H, int W)
{
    std::vector<std::vector<int>> path;
    int current_idx = goal_idx;
    while (current_idx != -1)
    {
        std::vector<int> coords = {current_idx % W, current_idx / W};
        path.push_back(coords);
        current_idx = parent_list[current_idx];
    }
    std::reverse(path.begin(), path.end());
    return path;
}

std::vector<std::vector<int>> solve_single(
    const std::vector<double> &pred
