//
// Created by zhaoyu on 2017/3/23.
//

#ifndef AP_AFFINITYPROPAGATION_H
#define AP_AFFINITYPROPAGATION_H

#include <functional>
#include <vector>

namespace AP {
    class AffinityPropagation {

    public:
        static void affinity_propagation(
                std::vector<int> &cluster_centers_indices,
                std::vector<int> &labels,
                std::vector< std::vector<double> > &S,
                int convergence_iter = 15,
                int max_iter = 200,
                double damping = 0.5
        );

    public:
        AffinityPropagation(double damping=0.5, int max_iter=200, int convergence_iter=15);
        void fit(const std::vector< std::vector<double> > &arr);

    public:
        double m_damping;
        int m_max_iter;
        int m_convergence_iter;
        std::vector<std::vector<double>> m_affinity_matrix;
        std::vector<int> m_cluster_centers_indices;
        std::vector<int> m_labels;
    };
}

#endif //AP_AFFINITYPROPAGATION_H
