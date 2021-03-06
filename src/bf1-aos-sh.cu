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

    Version BF1-AoS-Sh:
    - the input graph is stored as an adjacency list (Array of Structures),
    - the parallelization is done on the "inner cycle",
    - a shared memory buffer is used

    To compile:
    nvcc -arch=<cuda_capability> bf1-aos-sh.cu -o bf1-aos-sh

    To run:
    ./bf1-aos-sh < test/graph.txt > solution.txt
*/

#include "hpc.h"
#include "utils.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

typedef struct {
    // Number of neighbors
    unsigned int n_neighbors;

    // Array of indices of neighbor nodes
    unsigned int *neighbors;

    // Weights of outgoing arcs to neighbors
    unsigned int *weights;
} Node;

/*
    Reads a graph from stdin formatted as follows:
    first line: |number of nodes| |number of arcs| n
    each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

    The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

    This function returns a pointer to an array of |n_nodes| structures of type Node.
*/
Node* read_graph ( unsigned int *n_nodes, unsigned int *n_edges ) {
    /*
        |tmp| is necessary to read the third value of the first line, which is useless
    */
    unsigned int tmp;
    scanf("%u %u %u", n_nodes, n_edges, &tmp);

    Node *graph = (Node*) malloc((*n_nodes) * sizeof(Node));
    assert(graph);

    for(unsigned int i=0; i<*n_nodes; i++) {
        graph[i].n_neighbors = 0;
        graph[i].neighbors = NULL;
        graph[i].weights = NULL;
    }

    for(unsigned int i=0; i<*n_edges; i++) {
        unsigned int start_node, end_node, weight;
        float tmp;
        scanf("%u %u %f", &start_node, &end_node, &tmp);
        weight = (unsigned int)tmp;

        if(start_node >= *n_nodes || end_node >= *n_nodes) {
            fprintf(stderr, "ERROR at line %u: invalid node index\n\n", i+1);
            exit(EXIT_FAILURE);
        }

        graph[start_node].neighbors = (unsigned int*) realloc(graph[start_node].neighbors, (graph[start_node].n_neighbors+1)*sizeof(unsigned int*));
        assert(graph[start_node].neighbors);
        graph[start_node].weights = (unsigned int*) realloc(graph[start_node].weights, (graph[start_node].n_neighbors+1)*sizeof(unsigned int*));
        assert(graph[start_node].weights);
        graph[start_node].neighbors[graph[start_node].n_neighbors] = end_node;
        graph[start_node].weights[graph[start_node].n_neighbors] = weight;
        graph[start_node].n_neighbors++;
    }

    return graph;
}

/*
    CUDA kernel of Bellman-Ford's algorithm.
    A single block of |BLKDIM| threads executes a relax on each outgoing edge
    of each node.
*/
__global__ void cuda_bellman_ford (unsigned int n_nodes,
                                   Node* graph,
                                   unsigned int *distances) {
    if(blockIdx.x != 0) return;

    __shared__ unsigned int sh_buffer[2*BLKDIM];

    for(unsigned int node = 0; node < n_nodes; node++) {
        for(unsigned int idx = threadIdx.x; idx < graph[node].n_neighbors; idx += BLKDIM) {

            sh_buffer[2*threadIdx.x] = graph[node].neighbors[idx];
            sh_buffer[2*threadIdx.x+1] = graph[node].weights[idx];

            // relax the edge (u,v)
            const unsigned int u = node;
            const unsigned int v = sh_buffer[2*threadIdx.x];
            // overflow-safe check
            if(distances[v] > distances[u] && distances[v]-distances[u] > sh_buffer[2*threadIdx.x+1]) {
                distances[v] = distances[u] + sh_buffer[2*threadIdx.x+1];
            }
        }
        __syncthreads();
    }
}

/*
    This kernel is supposed to be executed from a single thread
    in a single block.
    Connects pointers to create an adjacency list on the device.
*/
__global__ void connect_pointers (Node *node,
                                  unsigned int *new_neighbors,
                                  unsigned int *new_weights) {
    const unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx != 0) return;
    node->neighbors = new_neighbors;
    node->weights = new_weights;
}

/*
    Executes the Bellman-Ford's algorithm on the graph |h_graph|.
    Returns a pointer to an array with |n_nodes| elements:
    each element of index |i| contains the shortest path distance from node
    |source| to node |i|.
*/
unsigned int* bellman_ford ( Node* h_graph, unsigned int n_nodes, unsigned int source ) {
    if(h_graph == NULL) return NULL;
    if(source >= n_nodes) {
        fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
        exit(EXIT_FAILURE);
    }

    size_t sz_distances = n_nodes * sizeof(unsigned int);
    size_t sz_graph = n_nodes * sizeof(Node);

    Node* d_graph;

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
    cudaSafeCall( cudaMalloc((void**)&d_graph, sz_graph) );
    cudaSafeCall( cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice) );

    // copying each node's neighbors
    for(unsigned int i=0; i<n_nodes; i++) {
        unsigned int *d_tmp_neighbors;
        unsigned int *d_tmp_weights;
        const unsigned int sz = h_graph[i].n_neighbors * sizeof(unsigned int);
        cudaSafeCall( cudaMalloc((void**)&d_tmp_neighbors, sz) );
        cudaSafeCall( cudaMemcpy(d_tmp_neighbors, h_graph[i].neighbors, sz, cudaMemcpyHostToDevice) );
        cudaSafeCall( cudaMalloc((void**)&d_tmp_weights, sz) );
        cudaSafeCall( cudaMemcpy(d_tmp_weights, h_graph[i].weights, sz, cudaMemcpyHostToDevice) );
        connect_pointers <<< 1, 1 >>> (&d_graph[i], d_tmp_neighbors, d_tmp_weights);
    }
    cudaCheckError();

    fprintf(stderr, "\n");

    // Computation
    for(unsigned int i=0; i<n_nodes-1; i++) {
        if(i%1000 == 0) fprintf(stderr, "%u / %u iterations completed\n", i, n_nodes-1);
        cuda_bellman_ford <<< 1, BLKDIM >>> (n_nodes, d_graph, d_distances);
        cudaCheckError();
    }

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_graph);
    cudaFree(d_distances);

    return h_distances;
}

void destroy_graph (unsigned int nodes, Node* graph) {
    for(unsigned int i=0; i<nodes; i++) {
        free(graph[i].neighbors);
        free(graph[i].weights);
    }
    free(graph);
}

int main ( void ) {

    Node *graph;
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

    print_ram_usage(nodes*sizeof(Node) + edges*2*sizeof(unsigned int));

    fprintf(stderr, "Computing Bellman-Ford...");
    compute_start = clock();
    result = bellman_ford(graph, nodes, 0);
    compute_end = clock();
    fprintf(stderr, "done\n\n");

    fprintf(stderr, "Dumping solution...");
    dump_solution(nodes, 0, result);
    fprintf(stderr, "done\n");

    destroy_graph(nodes, graph);
    free(result);

    program_end = clock();

    float total_seconds = (float)(program_end-program_start) / (float)CLOCKS_PER_SEC;
    float compute_seconds = (float)(compute_end-compute_start) / (float)CLOCKS_PER_SEC;

    fprintf(stderr, "\nTotal execution time: %.3f seconds\n", total_seconds);
    fprintf(stderr, "Actual execution time: %.3f seconds\n", compute_seconds);

    unsigned long long total_work = (unsigned long long) nodes * (unsigned long long) edges;
    double throughput = (double)total_work / (double)compute_seconds;
    fprintf(stderr, "\nThroughput: %.3e relax/second\n", throughput);

    return EXIT_SUCCESS;
}