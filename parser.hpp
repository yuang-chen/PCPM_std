#pragma once


#include <string>
#include <iostream>
#include <fstream>
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include "graph.hpp"


bool parseGraph(std::string filename, Graph& graph) {
    std::ifstream csr_file;
    csr_file.open(filename, std::ios::binary);
    if(!csr_file.is_open()) {
        std::cout << "cannot open csr file!" << std::endl;
        return false;
    }

    csr_file.read(reinterpret_cast<char*>(&graph.num_vertex), sizeof(unsigned));
    csr_file.read(reinterpret_cast<char*>(&graph.num_edges), sizeof(unsigned));

    std::vector<unsigned> local_row(graph.num_vertex);
    std::vector<unsigned> local_col(graph.num_edges);

    csr_file.read(reinterpret_cast<char*>(local_row.data()), graph.num_vertex * sizeof(unsigned));
    csr_file.read(reinterpret_cast<char*>(local_col.data()), graph.num_edges * sizeof(unsigned));

    local_row.push_back(graph.num_edges);
    graph.row_index = std::move(local_row);
    graph.col_index = std::move(local_col);
    csr_file.close();

    return true;
};

bool writeGraph(std::string filename, Graph& graph) {
    std::ofstream  txt_file(filename);
    if(!txt_file.is_open()) {
        std::cout << "cannot open txt file!" << std::endl;
        return false;
    }
// size_t len = 1;
//  txt_file.write(&len, sizeof(size_t));
    txt_file << graph.num_vertex << std::endl;
    txt_file << graph.num_edges << std::endl;

    for(int i = 0; i < graph.num_vertex + 1; i ++)
        txt_file << graph.row_index[i] << std::endl;

    for(int i = 0; i < graph.num_edges; i ++)
        txt_file << graph.col_index[i] << std::endl;

    txt_file.close();
};


