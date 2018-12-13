#include <iostream>
#include "AffinityPropagation.h"
#include <fstream>


std::vector<std::vector<double>> load_feature(const std::string& filename, const int row, const int col)
{
    std::ifstream is(filename);
    if (!is.is_open()) throw "file cannot be opened";

    std::vector<std::vector<double>> matrix(row, std::vector<double>(col, 0.));

    for (int fid = 0; fid < row; ++ fid) {
        for (int idx = 0; idx < col; ++ idx) {
            is >> matrix[fid][idx];
        }
    }

    return matrix;
}

int main() {
    auto affinity_matrix = load_feature("/home/terbe/R/affinity-propagation/affinity_matrix.txt", 115, 115);


    AP::AffinityPropagation ap;
    ap.fit(affinity_matrix);

    for (auto e: ap.m_labels) {
        std::cout << e << std::endl;
    }
	return 0;
}
