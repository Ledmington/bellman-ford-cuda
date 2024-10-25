/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF0-none-SoA-NoSh:
	- the input graph is stored as an array of weighted arcs (Structure of Arrays),
	- the parallelization is done on the "inner cycle",
	- no mutexes
	- no shared memory
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

/*
	CUDA kernel of Bellman-Ford's algorithm.
	Each thread executes a relax on a single edge in each kernel call.
*/
__global__ void cuda_bellman_ford(uint32_t n_edges, uint32_t* start_nodes, uint32_t* end_nodes, uint32_t* weights,
								  uint32_t* distances) {
	uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;

	if (idx < n_edges) {
		// relax the edge (u,v)
		const uint32_t u = start_nodes[idx];
		const uint32_t v = end_nodes[idx];
		// overflow-safe check
		if (distances[v] > distances[u] && distances[v] - distances[u] > weights[idx]) {
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
uint32_t* bellman_ford(Graph* h_graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (h_graph == NULL) {
		return NULL;
	}

	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
		exit(EXIT_FAILURE);
	}

	size_t sz_distances = n_nodes * sizeof(uint32_t);
	size_t sz = n_edges * sizeof(uint32_t);

	uint32_t* d_start_nodes;
	uint32_t* d_end_nodes;
	uint32_t* d_weights;

	uint32_t* d_distances;
	uint32_t* h_distances = (uint32_t*)malloc(sz_distances);
	assert(h_distances);

	for (uint32_t i = 0; i < n_nodes; i++) {
		h_distances[i] = UINT_MAX;
	}
	h_distances[source] = 0;

	// malloc and copy of the distances array
	cudaSafeCall(cudaMalloc((void**)&d_distances, sz_distances));
	cudaSafeCall(cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice));

	// malloc and copy of the graph
	cudaSafeCall(cudaMalloc((void**)&d_start_nodes, sz));
	cudaSafeCall(cudaMemcpy(d_start_nodes, h_graph->start_nodes, sz, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&d_end_nodes, sz));
	cudaSafeCall(cudaMemcpy(d_end_nodes, h_graph->end_nodes, sz, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void**)&d_weights, sz));
	cudaSafeCall(cudaMemcpy(d_weights, h_graph->weights, sz, cudaMemcpyHostToDevice));

	for (uint32_t i = 0; i < n_nodes - 1; i++) {
		cuda_bellman_ford<<<(n_edges + BLKDIM - 1) / BLKDIM, BLKDIM>>>(n_edges, d_start_nodes, d_end_nodes, d_weights,
																	   d_distances);
		cudaCheckError();
	}

	// copy-back of the result
	cudaSafeCall(cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost));

	// deallocation
	cudaFree(d_start_nodes);
	cudaFree(d_end_nodes);
	cudaFree(d_weights);
	cudaFree(d_distances);

	return h_distances;
}

int main(void) {
	Graph* graph;
	uint32_t nodes, edges;
	uint32_t* result;

	clock_t program_start, program_end, compute_start, compute_end;

	program_start = clock();

	fprintf(stderr, "Reading input graph...");
	graph = read_graph_soa(&nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage(3 * edges * sizeof(uint32_t));

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

	float total_seconds = (float)(program_end - program_start) / (float)CLOCKS_PER_SEC;
	float compute_seconds = (float)(compute_end - compute_start) / (float)CLOCKS_PER_SEC;

	fprintf(stderr, "\nTotal execution time: %.3f seconds\n", total_seconds);
	fprintf(stderr, "Actual execution time: %.3f seconds\n", compute_seconds);

	unsigned long long total_work = (unsigned long long)nodes * (unsigned long long)edges;
	double throughput = (double)total_work / (double)compute_seconds;
	fprintf(stderr, "\nThroughput: %.3e relax/second\n", throughput);

	return EXIT_SUCCESS;
}