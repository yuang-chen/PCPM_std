#pragma once

#include <iostream>
#include <stdio.h>
#include <stdlib.h>
#include <pthread.h>
#include <boost/timer/timer.hpp>

#include <algorithm>
#include <omp.h>
#include <assert.h>
#include <immintrin.h>
#include <assert.h>

#include "graph.hpp"
#include "vec2d.hpp"
#include "global.hpp"


enum MASK {
    MAX_NEG = 0x80000000,
    MAX_POS = 0x7fffffff,
    MAX_UINT = 0xffffffff  
};


class Partitioner {
    // (num_clusters + 1) X num_clusters 
    std::vector<unsigned> dstver_offset;  
    std::vector<unsigned> buffer_offset; 
    // num_clusters X varied length 
    std::vector<float*> buffer_ptr;
    std::vector<unsigned*> dstver_ptr;

    std::vector<Cluster> clusters;

public:

    Partitioner(unsigned num_clusters) {

        clusters = std::move(std::vector<Cluster>(num_clusters));
        // offset vector size: (num_clusters + 1) X num_clusters 
     //   dstver_offset = IntArray2d(boost::extents[3 + 1][3]);
        dstver_offset = std::move(std::vector<unsigned>((num_clusters + 1) * num_clusters, 0));
        buffer_offset = std::move(std::vector<unsigned>((num_clusters + 1) * num_clusters, 0));
      
        buffer_ptr = std::move(std::vector<float*>(num_clusters * num_clusters, nullptr));
        dstver_ptr = std::move(std::vector<unsigned*>(num_clusters * num_clusters, nullptr));
      //  vertex_range = std::vector<std::pair<unsigned, unsigned>>(num_clusters);
    }
    // split the original graph into clsgraphs and belonging partData
    void split(const Graph& graph) {
        unsigned num_clusters = params::num_clusters;
        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for(int n = 0; n < num_clusters; n++) {
            int start = n * params::cluster_size;
            int end = (n + 1) * params::cluster_size > graph.num_vertex?
                       graph.num_vertex: (n + 1) * params::cluster_size;
        
            Cluster& cls = clusters[n];
            cls.range = std::pair<unsigned, unsigned>(start, end); 
            cls.act_row_index = std::move(std::vector<unsigned>(num_clusters + 1, 0));

            /////////////////////////////////////////
            // row_index & col_index lists that,
            // how many outgoing fat edges in next cluster would update other clsgraphs, including itself
            /////////////////////////////////////////
            /////////////////////////////////////////
            // dstver_offset: excatly how many thin edges in nextcls. would update other cls.
            // buffer_offset: it also records the number of fat edges
            /////////////////////////////////////////
            // compute the clsgraph.row_index
            ///////////////////////////////////////

            int nextPart, prevPart;
            for(int i = start; i < end; i++) {
                prevPart = num_clusters + 1;
                for(int j = graph.row_index[i]; j < graph.row_index[i+1]; j++) {
                    nextPart = graph.col_index[j] >> params::cluster_offset;
                    dstver_offset[at(n+1, nextPart)]++;
                    if(nextPart == prevPart)
                        continue;
                    cls.act_row_index[nextPart + 1]++;
                    prevPart = nextPart;
                }
            }

            for(int i = 0; i < num_clusters; i++)
                buffer_offset[at(n+1, i)] = cls.act_row_index[i + 1];
            
            for(int i = 0; i < num_clusters; i++)
                cls.act_row_index[i+1] += cls.act_row_index[i];
           
            cls.num_act_edges = cls.act_row_index[num_clusters];            
            cls.act_col_index = std::move(std::vector<unsigned>(cls.num_act_edges, 0));
            /////////////////////////////////////////
            // compute the clsgraph.col_index
            ///////////////////////////////////////
            std::vector<unsigned> tmp_offset(num_clusters,0);
            
            for(int i = start; i < end; i++) {
                prevPart = num_clusters + 1;
                for(int j = graph.row_index[i]; j < graph.row_index[i+1]; j++) {
                    nextPart = graph.col_index[j] >> params::cluster_offset;
                    if(nextPart == prevPart)
                        continue;
                    cls.act_col_index[cls.act_row_index[nextPart] + tmp_offset[nextPart]++] = i;
                    prevPart = nextPart;
                }
            }  
        }
                //////////////////////////////////
        // after tranpose
        // accumulate in column, which counts the incoming fat edges, including itself
        // so the last row -- **_offset[num_clusters] presents the total number of incoming fat edges
        // critical for gathering phase
        ////////////////////////////////
        for(int i = 0; i < num_clusters; i++) {
            for(int j = 0; j < num_clusters; j++) {
                buffer_offset[at(j + 1, i)] += buffer_offset[at(j, i)];
                dstver_offset[at(j + 1, i)] += dstver_offset[at(j, i)];
            }
        }
    }
   
    void setCluster(const Graph& graph) {
        ////////////////////////////////////////////////////
        // here we initialize the updates and dstverx vectors, 
        // updates/dstverx: num_clusters X num_
        // their value will be assigned later
        //////////////////////////////////////
        unsigned num_clusters = params::num_clusters;

        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for(int n = 0; n < num_clusters; n++) {
            auto& cls = clusters[n];
            cls.buffer = std::move(std::vector<float>(buffer_offset[at(num_clusters, n)]));
            cls.dstver = std::move(std::vector<unsigned>(dstver_offset[at(num_clusters, n)] + 1));
            cls.dstver[dstver_offset[at(num_clusters, n)]] = MAX_NEG;
        }
        /*
        for(auto cls_it = clusters.begin(); cls_it != clusters.end(); ++cls_it) {
            unsigned i = cls_it - clusters.begin();
            cls_it->buffer = std::move(std::vector<float>(buffer_offset[at(num_clusters, i)]));
            cls_it->dstver = std::move(std::vector<unsigned>(dstver_offset[at(num_clusters, i)] + 1));
            cls_it->dstver[dstver_offset[at(num_clusters, i)]] = MAX_NEG;
        }*/

        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for(unsigned i = 0; i < num_clusters; i++) 
            for(unsigned j = 0; j < num_clusters; j++) {
                buffer_ptr[at(i, j)] = clusters[j].buffer.data() + buffer_offset[at(i, j)];
                dstver_ptr[at(i, j)] = clusters[j].dstver.data() + dstver_offset[at(i, j)];
            }
        ///////////////////////////////////////
        /// here we assign the values of dstverx & updates
        //////////////////////////////////////
       
        //for(auto cls_it = clusters.begin(); cls_it != clusters.end(); ++cls_it) {
        #pragma omp parallel for schedule(static) num_threads(NUM_THREADS)
        for(int n = 0; n < num_clusters; n++) { 
            auto& cls = clusters[n];
            unsigned next_ver = 0;
            unsigned next_cls = 0;
            unsigned prev_cls = 0;

            std::vector<unsigned> count_buffer(num_clusters, 0);
            std::vector<unsigned> count_dstver(num_clusters, 0);

            for(auto i = cls.range.first; i < cls.range.second; i++) {
                prev_cls = num_clusters;
                for(auto j = graph.row_index[i]; j < graph.row_index[i+1]; ++j) {
                    next_ver = graph.col_index[j];
                    next_cls = next_ver >> params::cluster_offset;
                    if(prev_cls != next_cls) {
                        buffer_ptr[at(n, next_cls)][count_buffer[next_cls]++] = graph.attr[i];
                        next_ver |= MASK::MAX_NEG; 
                        prev_cls = next_cls;
                    }
                    dstver_ptr[at(n, next_cls)][count_dstver[next_cls]++] = next_ver;
                }
            }
        }
    }
    ///////////////////////
    // gather-pull
    //////////////////////
    void gather(std::vector<float>& attr, const std::vector<unsigned>& out_degree) {
        
       #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
        for(int n = 0; n < params::num_clusters; n++) { 
           // double wtime = omp_get_wtime();
            auto& cls = clusters[n];
            unsigned start_vertex = cls.range.first;
            unsigned end_vertex = cls.range.second;
        
            for(unsigned i = start_vertex; i < end_vertex; i++)
                attr[i] = 0.0;

          //  unsigned update_index = MAX_UINT;
            unsigned next_ver = 0;
            unsigned index = MASK::MAX_UINT;
            for(unsigned i = 0; i < cls.dstver.size() - 1; i++) {
                next_ver = cls.dstver[i];
                index +=  (next_ver >> 31);
                // the next_ver & MAX_POS are limited between start_vertex and end_vertex
                attr[next_ver & MASK::MAX_POS] += cls.buffer[index];
            }

            /////////////////////////////////
            // damping factor for page rank
            ////////////////////////////////
            for(unsigned i = start_vertex; i < end_vertex; i++) {
                attr[i] = params::damping + (1 - params::damping) * attr[i];
                if(out_degree[i] > 0)
                    attr[i] = attr[i] / out_degree[i];
            }
         //   wtime = omp_get_wtime() - wtime;
         //   printf( "Time taken by thread %d is %f\n", omp_get_thread_num(), wtime);
        }
       
    }

    void scatter(std::vector<float>& attr) {
        #pragma omp parallel for schedule(dynamic) num_threads(NUM_THREADS)
        for(int n = 0; n < params::num_clusters; n++) { 
            auto& cls = clusters[n];
            for(unsigned next_cls = 0; next_cls < params::num_clusters; ++next_cls) {
                unsigned index = 0;
                unsigned next_ver = 0;
                for(auto j = cls.act_row_index[next_cls]; j < cls.act_row_index[next_cls + 1]; j++) {
                    next_ver = cls.act_col_index[j];
                    // the next_ver is always is in sequence so the access efficiency is quite good
                    buffer_ptr[at(n, next_cls)][index++] = attr[next_ver];
                }
            }
        }
    }

    void printCluster(bool all = false) {
        unsigned total_act_edges = 0;

        
        for(auto cls_it = clusters.begin(); cls_it != clusters.end(); ++cls_it) {
            if(all == true)
                std::cout << "cluster " << (cls_it - clusters.begin() + 1)
                        << " | num of fat edges: " << cls_it->num_act_edges
                        << " | size of buffer vector: " << cls_it->buffer.size() 
                        << " | size of dstver vector: " << cls_it->dstver.size()  << std::endl;
            total_act_edges += cls_it->num_act_edges;
        }
        std::cout << "total_act_edges: " << total_act_edges << std::endl;
    }
};



