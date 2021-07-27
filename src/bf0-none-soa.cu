/*
    CUDA implementation of the Bellman-Ford's algorithm
    Copyright (C) 2021  Filippo Barbari

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
*/
/*
    CUDA implementation of the Bellman-Ford's algorithm.

    Version BF0-none-SoA:
    - the input graph is stored as an array of weighted arcs (Structure of Arrays),
    - the parallelization is done on the "inner cycle",
    - no mutexes

    To compile:
    nvcc -arch=<cuda_capability> bf0-none.cu -o bf0-none

    To run:
    ./bf0-none < test/graph.txt > solution.txt
*/

#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

typedef struct {
    // The indexes of the source nodes of each edge
    unsigned int* start_nodes;

    // The indexes of the destination nodes of each edge
    unsigned int* end_nodes;

    // The weights assigned to each edge
    unsigned int* weights;
} Graph;

/*
    Reads a graph from stdin formatted as follows:
    first line: |number of nodes| |number of arcs| n
    each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

    The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

    This function returns a pointer to a Graph structure.
*/
Graph* read_graph ( unsigned int *n_nodes, unsigned int *n_edges ) {
    /*
        |tmp| is necessary to read the third value of the first line, which is useless
    */
    unsigned int tmp;
    scanf("%u %u %u", n_nodes, n_edges, &tmp);

    Graph *graph = (Graph*) malloc(sizeof(Graph));
    assert(graph);

    graph->start_nodes = (unsigned int*) malloc((*n_edges) * sizeof(unsigned int));
    assert(graph->start_nodes);
    graph->end_nodes = (unsigned int*) malloc((*n_edges) * sizeof(unsigned int));
    assert(graph->end_nodes);
    graph->weights = (unsigned int*) malloc((*n_edges) * sizeof(unsigned int));
    assert(graph->weights);

    for(unsigned int i=0; i<*n_edges; i++) {
        float tmp;
        scanf("%u %u %f", &graph->start_nodes[i], &graph->end_nodes[i], &tmp);
        graph->weights[i] = (unsigned int)tmp;

        if(graph->start_nodes[i] >= *n_nodes || graph->end_nodes[i] >= *n_nodes) {
            fprintf(stderr, "ERROR at line %u: invalid node index\n\n", i+1);
            exit(EXIT_FAILURE);
        }
    }

    return graph;
}

/*
    Dumps the solution on stdout.

    Output is formatted as follows:

    number_of_nodes
    source_node
    node_0 distance_to_node_0
    node_1 distance_to_node_1
    node_2 distance_to_node_2
    ...
*/
void dump_solution (unsigned int n_nodes, unsigned int source, unsigned int *dist) {
    printf("%u\n%u\n", n_nodes, source);

    for(unsigned int i=0; i<n_nodes; i++) {
        printf("%u %u\n", i, dist[i]);
    }
}

/*
    CUDA kernel of Bellman-Ford's algorithm.
    Each thread executes a relax on a single edge in each kernel call.
*/
__global__ void cuda_bellman_ford (unsigned int n_nodes,
                                   unsigned int n_edges,
                                   unsigned int* start_nodes,
                                   unsigned int* end_nodes,
                                   unsigned int* weights,
                                   unsigned int *distances) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_edges) {
        // relax the edge (u,v)
        const unsigned int u = start_nodes[idx];
        const unsigned int v = end_nodes[idx];
        // overflow-safe check
        if(distances[v] > distances[u] && distances[v]-distances[u] > weights[idx]) {
            distances[v] = distances[u] + weights[idx];
        }
    }
}

/*
    Executes the Bellman-Ford's algorithm on the graph |h_graph|.
    Returns a pointer to an array with |n_nodes| elements:
    each element of index |i| contains the shortest path distance from node
    |source| to node |i|.
*/
unsigned int* bellman_ford ( Graph* h_graph, unsigned int n_nodes, unsigned int n_edges, unsigned int source ) {
    if(h_graph == NULL) return NULL;
    if(source >= n_nodes) {
        fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
        exit(EXIT_FAILURE);
    }

    size_t sz_distances   = n_nodes * sizeof(unsigned int);
    size_t sz = n_edges * sizeof(unsigned int);

    unsigned int* d_start_nodes;
    unsigned int* d_end_nodes;
    unsigned int* d_weights;

    unsigned int *d_distances;
    unsigned int *h_distances = (unsigned int*) malloc(sz_distances);
    assert(h_distances);

    for(unsigned int i=0; i<n_nodes; i++) {
        h_distances[i] = UINT_MAX;
    }
    h_distances[source] = 0;

    // malloc and copy of the distances array
    cudaSafeCall( cudaMalloc((void**)&d_distances, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice) );

    // malloc and copy of the graph
    cudaSafeCall( cudaMalloc((void**)&d_start_nodes, sz) );
    cudaSafeCall( cudaMemcpy(d_start_nodes, h_graph->start_nodes, sz, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc((void**)&d_end_nodes, sz) );
    cudaSafeCall( cudaMemcpy(d_end_nodes, h_graph->end_nodes, sz, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc((void**)&d_weights, sz) );
    cudaSafeCall( cudaMemcpy(d_weights, h_graph->weights, sz, cudaMemcpyHostToDevice) );

    for(unsigned int i=0; i<n_nodes-1; i++) {
        cuda_bellman_ford <<< (n_edges+BLKDIM-1) / BLKDIM, BLKDIM >>> (n_nodes, n_edges, d_start_nodes, d_end_nodes, d_weights, d_distances);
        cudaCheckError();
    }

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_start_nodes);
    cudaFree(d_end_nodes);
    cudaFree(d_weights);
    cudaFree(d_distances);

    return h_distances;
}

int main ( void ) {

    Graph *graph;
    unsigned int nodes, edges;
    unsigned int *result;

    clock_t program_start, program_end, compute_start, compute_end;

    program_start = clock();

    fprintf(stderr, "Reading input graph...");
    graph = read_graph(&nodes, &edges);
    fprintf(stderr, "done\n");

    fprintf(stderr, "\nGraph data:\n");
    fprintf(stderr, " %7u nodes\n", nodes);
    fprintf(stderr, " %7u arcs\n", edges);

    float ram_usage = (float)(3 * edges * sizeof(unsigned int));
    if(ram_usage < 1024.0f) {
        fprintf(stderr, " %.3f bytes of RAM used\n\n", ram_usage);
    }
    else if(ram_usage < 1024.0f*1024.0f) {
        fprintf(stderr, " %.3f KBytes of RAM used\n\n", ram_usage/1024.0f);
    }
    else {
        fprintf(stderr, " %.3f MBytes of RAM used\n\n", ram_usage/(1024.0f*1024.0f));
    }

    fprintf(stderr, "Computing Bellman-Ford...");
    compute_start = clock();
    result = bellman_ford(graph, nodes, edges, 0);
    compute_end = clock();
    fprintf(stderr, "done\n\n");

    fprintf(stderr, "Dumping solution...");
    dump_solution(nodes, 0, result);
    fprintf(stderr, "done\n");

    free(graph->start_nodes);
    free(graph->end_nodes);
    free(graph->weights);
    free(graph);
    free(result);

    program_end = clock();

    fprintf(stderr, "\nTotal execution time: %.3f seconds\n", (float)(program_end-program_start) / (float)CLOCKS_PER_SEC);
    fprintf(stderr, "Actual execution time: %.3f seconds\n", (float)(compute_end-compute_start) / (float)CLOCKS_PER_SEC);


    return EXIT_SUCCESS;
}