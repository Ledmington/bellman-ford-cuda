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

    Version BF0-mutex-AoS:
    - the input graph is stored as an array of weighted arcs (Array of Structures),
    - the parallelization is done on the "inner cycle",
    - an atomic operation is used for the update of distances

    To compile:
    nvcc -arch=<cuda_capability> bf0-mutex-aos.cu -o bf0-mutex-aos

    To run:
    ./bf0-mutex-aos < test/graph.txt > solution.txt
*/

#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

typedef struct {
    // The index of the source node of the edge
    unsigned int start_node;

    // The index of the destination node of the edge
    unsigned int end_node;

    // The weight assigned to the edge
    float weight;
} Edge;

/*
    Reads a graph from stdin formatted as follows:
    first line: |number of nodes| |number of arcs| n
    each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

    The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

    This function returns a pointer to an array of |n_edges| structures of type Edge.
*/
Edge* read_graph ( unsigned int *n_nodes, unsigned int *n_edges ) {
    /*
        |tmp| is necessary to read the third value of the first line, which is useless
    */
    unsigned int tmp;
    scanf("%u %u %u", n_nodes, n_edges, &tmp);

    Edge *graph = (Edge*) malloc(*n_edges * sizeof(Edge));
    assert(graph);

    for(unsigned int i=0; i<*n_edges; i++) {
        scanf("%u %u %f", &graph[i].start_node, &graph[i].end_node, &graph[i].weight);

        if(graph[i].start_node >= *n_nodes || graph[i].end_node >= *n_nodes) {
            fprintf(stderr, "ERROR at line %u: invalid node index.\n\n", i+1);
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
void dump_solution (unsigned int n_nodes, unsigned int source, float *dist) {
    printf("%u\n%u\n", n_nodes, source);

    for(unsigned int i=0; i<n_nodes; i++) {
        printf("%u", i);
        if(isinf(dist[i])) {
            printf(" %u\n", UINT_MAX);
        }
        else {
            printf(" %u\n", (unsigned int)dist[i]);
        }
    }
}

/*
    CUDA kernel of Bellman-Ford's algorithm.
    Each thread executes a relax on a single edge in each kernel call.
*/
__global__ void cuda_bellman_ford (unsigned int n_edges,
                                   Edge* graph,
                                   float *distances) {
    union {
        float vf;
        int vi;
    } oldval, newval;

    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_edges) {
        // relax the edge (u,v)
        const unsigned int u = graph[idx].start_node;
        const unsigned int v = graph[idx].end_node;

        if(distances[u] + graph[idx].weight < distances[v]) {
            do {
                oldval.vf = distances[v];
                newval.vf = distances[u] + graph[idx].weight;
                newval.vf = fminf(oldval.vf, newval.vf);
            } while( atomicCAS((int*)&distances[v], oldval.vi, newval.vi) != newval.vi );
        }
    }
}

/*
    Executes the Bellman-Ford's algorithm on the graph |h_graph|.
    Returns a pointer to an array with |n_nodes| elements:
    each element of index |i| contains the shortest path distance from node
    |source| to node |i|.
*/
float* bellman_ford ( Edge* h_graph, unsigned int n_nodes, unsigned int n_edges, unsigned int source ) {
    if(h_graph == NULL) return NULL;
    if(source >= n_nodes) {
        fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
        exit(EXIT_FAILURE);
    }

    size_t sz_distances = n_nodes * sizeof(float);
    size_t sz_graph = n_edges * sizeof(Edge);

    Edge* d_graph;

    float *d_distances;
    float *h_distances = (float*) malloc(sz_distances);
    assert(h_distances);

    for(unsigned int i=0; i<n_nodes; i++) {
        h_distances[i] = HUGE_VAL;
    }
    h_distances[source] = 0.0f;

    // malloc and copy of the distances array
    cudaSafeCall( cudaMalloc((void**)&d_distances, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice) );

    // malloc and copy of the graph
    cudaSafeCall( cudaMalloc((void**)&d_graph, sz_graph) );
    cudaSafeCall( cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice) );

    for(unsigned int i=0; i<n_nodes-1; i++) {
        cuda_bellman_ford <<< (n_edges+BLKDIM-1) / BLKDIM, BLKDIM >>> (n_edges, d_graph, d_distances);
        cudaCheckError();
    }

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_graph);
    cudaFree(d_distances);

    return h_distances;
}

int main ( void ) {

    Edge *graph;
    unsigned int nodes, edges;
    float *result;

    clock_t program_start, program_end, compute_start, compute_end;

    program_start = clock();

    fprintf(stderr, "Reading input graph...");
    graph = read_graph(&nodes, &edges);
    fprintf(stderr, "done\n");

    fprintf(stderr, "\nGraph data:\n");
    fprintf(stderr, " %7u nodes\n", nodes);
    fprintf(stderr, " %7u arcs\n", edges);

    float ram_usage = (float)(sizeof(Edge)*edges);
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

    free(graph);
    free(result);

    program_end = clock();

    fprintf(stderr, "\nTotal execution time: %.3f seconds\n", (float)(program_end-program_start) / (float)CLOCKS_PER_SEC);
    fprintf(stderr, "Actual execution time: %.3f seconds\n", (float)(compute_end-compute_start) / (float)CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}