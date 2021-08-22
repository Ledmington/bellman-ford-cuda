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

    Version BF1-SoA-NoSh:
    - the input graph is stored as an adjacency list (Structure of Arrays),
    - the parallelization is done on the "inner cycle",
    - no shared memory is used

    To compile:
    nvcc -arch=<cuda_capability> bf1-soa-nosh.cu -o bf1-soa-nosh

    To run:
    ./bf1-soa-nosh < test/graph.txt > solution.txt
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

typedef struct {
    // start_indices[i] is the index of the first neighbor of node |i|.
    unsigned int *start_indices;

    // Number of neighbors of each node
    unsigned int *n_neighbors;

    // Indices of neighbor nodes
    unsigned int *neighbors;

    // Weights of outgoing arcs to neighbors
    unsigned int *weights;
} Graph;

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
    Converts the given array of |Node|s into a |Graph| structure (SoA).
*/
Graph* convert_to_soa (Node* list_of_nodes, unsigned int n_nodes, unsigned int n_edges) {
    Graph *graph = (Graph*) malloc(sizeof(Graph));
    assert(graph);

    graph->start_indices = (unsigned int*) malloc(n_nodes * sizeof(unsigned int));
    assert(graph->start_indices);
    graph->n_neighbors = (unsigned int*) malloc(n_nodes * sizeof(unsigned int));
    assert(graph->n_neighbors);
    graph->neighbors = (unsigned int*) malloc(n_edges * sizeof(unsigned int));
    assert(graph->neighbors);
    graph->weights = (unsigned int*) malloc(n_edges * sizeof(unsigned int));
    assert(graph->weights);

    unsigned int start_idx = 0;
    for(unsigned int i=0; i<n_nodes; i++) {
        graph->start_indices[i] = start_idx;
        graph->n_neighbors[i] = list_of_nodes[i].n_neighbors;

        const unsigned int sz = graph->n_neighbors[i] * sizeof(unsigned int);

        // Copying neighbors
        memcpy(&graph->neighbors[start_idx], list_of_nodes[i].neighbors, sz);

        // Copying weights
        memcpy(&graph->weights[start_idx], list_of_nodes[i].weights, sz);

        start_idx += graph->n_neighbors[i];
    }

    return graph;
}

/*
    CUDA kernel of Bellman-Ford's algorithm.
    A single block of |BLKDIM| threads executes a relax on each outgoing edge
    of each node.
*/
__global__ void cuda_bellman_ford (unsigned int n_nodes,
                                   unsigned int *start_indices,
                                   unsigned int *n_neighbors,
                                   unsigned int *neighbors,
                                   unsigned int *weights,
                                   unsigned int *distances) {
    if(blockIdx.x != 0) return;

    for(unsigned int node = 0; node < n_nodes; node++) {
        for(unsigned int idx = threadIdx.x; idx < n_neighbors[node]; idx += BLKDIM) {
            // relax the edge (u,v)
            const unsigned int u = node;
            const unsigned int v = neighbors[start_indices[node] + idx];
            // overflow-safe check
            if(distances[v] > distances[u] && distances[v]-distances[u] > weights[start_indices[node] + idx]) {
                distances[v] = distances[u] + weights[start_indices[node] + idx];
            }
        }
        __syncthreads();
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

    size_t sz_distances = n_nodes * sizeof(unsigned int);
    size_t sz_neighbors = n_edges * sizeof(unsigned int);

    unsigned int *d_start_indices;
    unsigned int *d_n_neighbors;
    unsigned int *d_neighbors;
    unsigned int *d_weights;

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
    cudaSafeCall( cudaMalloc((void**)&d_start_indices, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_start_indices, h_graph->start_indices, sz_distances, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc((void**)&d_n_neighbors, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_n_neighbors, h_graph->n_neighbors, sz_distances, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc((void**)&d_neighbors, sz_neighbors) );
    cudaSafeCall( cudaMemcpy(d_neighbors, h_graph->neighbors, sz_neighbors, cudaMemcpyHostToDevice) );
    cudaSafeCall( cudaMalloc((void**)&d_weights, sz_neighbors) );
    cudaSafeCall( cudaMemcpy(d_weights, h_graph->weights, sz_neighbors, cudaMemcpyHostToDevice) );

    fprintf(stderr, "\n");

    // Computation
    for(unsigned int i=0; i<n_nodes-1; i++) {
        if(i%1000 == 0) fprintf(stderr, "%u / %u iterations completed\n", i, n_nodes-1);
        cuda_bellman_ford <<< 1, BLKDIM >>> (n_nodes, d_start_indices, d_n_neighbors, d_neighbors, d_weights, d_distances);
        cudaCheckError();
    }

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_start_indices);
    cudaFree(d_n_neighbors);
    cudaFree(d_neighbors);
    cudaFree(d_weights);
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

    Node *list_of_nodes;
    Graph* graph;
    unsigned int nodes, edges;
    unsigned int *result;

    clock_t program_start, program_end, compute_start, compute_end;

    program_start = clock();

    fprintf(stderr, "Reading input graph...");
    list_of_nodes = read_graph(&nodes, &edges);
    fprintf(stderr, "done\n");

    graph = convert_to_soa(list_of_nodes, nodes, edges);
    destroy_graph(nodes, list_of_nodes);

    fprintf(stderr, "\nGraph data:\n");
    fprintf(stderr, " %7u nodes\n", nodes);
    fprintf(stderr, " %7u arcs\n", edges);

    print_ram_usage((2*nodes + 2*edges) * sizeof(unsigned int));

    fprintf(stderr, "Computing Bellman-Ford...");
    compute_start = clock();
    result = bellman_ford(graph, nodes, edges, 0);
    compute_end = clock();
    fprintf(stderr, "done\n\n");

    fprintf(stderr, "Dumping solution...");
    dump_solution(nodes, 0, result);
    fprintf(stderr, "done\n");

    free(graph->start_indices);
    free(graph->n_neighbors);
    free(graph->neighbors);
    free(graph->weights);
    free(graph);
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