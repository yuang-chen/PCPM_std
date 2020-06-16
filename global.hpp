#pragma once

#include <cmath>

namespace params {
    float damping = 0.15;
    unsigned int cluster_size = (256 * 1024)/sizeof(float);//(256*1024)/sizeof(float); //512kB
    unsigned int cluster_offset = (unsigned)log2((float)cluster_size); 
    unsigned int num_threads = 1;
    unsigned int num_clusters = 10000000/cluster_size;
    unsigned int num_iters = 100;
};

const int NUM_THREADS = omp_get_max_threads();


// 2^13 < 10480 < 2^14. 10480 is the size of L3 cache