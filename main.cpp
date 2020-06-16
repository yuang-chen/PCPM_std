/**
 * Author: Kartik Lakhotia
 * Email id: klakhoti@usc.edu
 * Date: 27-Jul-2017
 *
 * This code implements work optimized propagation blocking with
 * transposed bin graph to reduce cache misses in scatter
 */
#include <pthread.h>
#include <time.h>
#include "graph.hpp"
#include "partition.hpp"
#include "parser.hpp"
#include <boost/timer/timer.hpp>
#include "vec2d.hpp"
#include "global.hpp"

using namespace std;
using namespace boost::timer;
//////////////////////////////////////////
//main function
//////////////////////////////////////////
int main(int argc, char** argv)
{
    string data_file = argv[1];

    if (argc >= 3)
        params::num_threads = (unsigned int)atoi(argv[2]);
    if (argc >= 4)
        params::num_iters = (unsigned int)atoi(argv[3]);
    if ((argc < 2) || (argc > 4))
    {
        printf("Usage : %s <filename> <numThreads(optional)> <#iterations(optional)> \n", argv[0]);
        exit(1);
    }
   
    omp_set_num_threads(NUM_THREADS);
    // graph object
    Graph graph;

    /**************************************************
     Compute the preprocessing time
     *************************************************/
    cpu_timer timer;
    cpu_times times;

    if(parseGraph(data_file, graph))
        cout << times.wall/(1e9) << "s: parse done! " << endl;
    //////////////////////////////////////////
    // read csr file
    //////////////////////////////////////////
   
    graph.printGraph();

    params::num_clusters = (graph.num_vertex-1)/params::cluster_size + 1;
    cout << "number of subgraphs: " << params::num_clusters << endl;
    //////////////////////////////////////////
    // output Degree array
    //////////////////////////////////////////
    graph.computeOutDegree();

    //////////////////////////////////////////
    // initialize page rank attribute to 1/degree
    //////////////////////////////////////////
    graph.initAttribute();
    times = timer.elapsed();
    cout << times.wall/(1e9) << "s: initial attributes are assigned "  << endl;

   ///////////////////////////////
    // initialize the partitioner
    /////////////////////////////
    Partitioner partitioner(params::num_clusters);
    //////////////////////////////////////////
    // split the graph into clusters
    //////////////////////////////////////////
    partitioner.split(graph);
    
    times = timer.elapsed();
    cout << times.wall/(1e9) << "s: split the graph into clusters "  << endl;

    //////////////////////////////////////////
    // construct the clusters
    // distribute the attributes of graph to cluters` local buffers
    //////////////////////////////////////////
    partitioner.setCluster(graph);

    times = timer.elapsed();
    cout << times.wall/(1e9) << "s: clusters are built "  << endl;


    partitioner.gather(graph.attr, graph.out_degree);
   
    times = timer.elapsed();
    cout << times.wall/(1e9) << "s: initial gathering is done -- preprossing finished "  << endl;

    timer.start();
    //////////////////////////////////////////
    // iterate for 10 epochs
    //////////////////////////////////////////
    int iter = 0;
    while(iter < params::num_iters) {
    
        partitioner.scatter(graph.attr);
     //   times = timer.elapsed();
     //   cout << times.wall/(1e9) << "s: "<< iter << " iteration | scattering"  << endl;

        partitioner.gather(graph.attr, graph.out_degree);
     //   times = timer.elapsed();
     //   cout << times.wall/(1e9) << "s: "<< iter << " iteration | gathering"  << endl;
        iter++;
    }
    times = timer.elapsed();
    cout << argv[1] << ", processing time: " << times.wall/(1e9) << endl;
    //////////////////////////////////////////
    // Compute the processing time
    //////////////////////////////////////////
    partitioner.printCluster(false);
    
}
