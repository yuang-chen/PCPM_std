#pragma once

#include <iostream>
#include <vector>
#include <boost/multi_array.hpp>
#include <boost/smart_ptr.hpp>

#include "global.hpp"

template <typename T>
using array2d = typename boost::multi_array<T, 2>;

typedef boost::multi_array<unsigned,2> IntArray2d; 

template <typename T>
using vector2d = typename std::vector<std::vector<T>>;


inline unsigned at(unsigned row, unsigned col) {
    return row * params::num_clusters + col;
}

