/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF1-SoA-NoSh:
	- the input graph is stored as an adjacency list (Structure of Arrays),
	- the parallelization is done on the "inner cycle",
	- no shared memory is used
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

/*
	Converts the given array of |Node|s into a |Graph| structure (SoA).
*/
Graph_soa *convert_to_soa(Node *list_of_nodes, uint32_t n_nodes, uint32_t n_edges) {
	Graph_soa *graph = (Graph_soa *)malloc(sizeof(Graph_soa));
	assert(graph);

	graph->start_indices = (uint32_t *)malloc(n_nodes * sizeof(uint32_t));
	assert(graph->start_indices);
	graph->n_neighbors = (uint32_t *)malloc(n_nodes * sizeof(uint32_t));
	assert(graph->n_neighbors);
	graph->neighbors = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
	assert(graph->neighbors);
	graph->weights = (uint32_t *)malloc(n_edges * sizeof(uint32_t));
	assert(graph->weights);

	uint32_t start_idx = 0;
	for (uint32_t i = 0; i < n_nodes; i++) {
		graph->start_indices[i] = start_idx;
		graph->n_neighbors[i] = list_of_nodes[i].n_neighbors;

		const uint32_t sz = graph->n_neighbors[i] * sizeof(uint32_t);

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
__global__ void cuda_bellman_ford(uint32_t n_nodes, uint32_t *start_indices, uint32_t *n_neighbors, uint32_t *neighbors,
								  uint32_t *weights, uint32_t *distances) {
	if (blockIdx.x != 0) {
		return;
	}

	for (uint32_t node = 0; node < n_nodes; node++) {
		for (uint32_t idx = threadIdx.x; idx < n_neighbors[node]; idx += BLKDIM) {
			// relax the edge (u,v)
			const uint32_t u = node;
			const uint32_t v = neighbors[start_indices[node] + idx];
			// overflow-safe check
			if (distances[v] > distances[u] && distances[v] - distances[u] > weights[start_indices[node] + idx]) {
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
uint32_t *bellman_ford(Graph_soa *h_graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (h_graph == NULL) {
		return NULL;
	}

	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
		exit(EXIT_FAILURE);
	}

	size_t sz_distances = n_nodes * sizeof(uint32_t);
	size_t sz_neighbors = n_edges * sizeof(uint32_t);

	uint32_t *d_start_indices;
	uint32_t *d_n_neighbors;
	uint32_t *d_neighbors;
	uint32_t *d_weights;

	uint32_t *d_distances;
	uint32_t *h_distances = (uint32_t *)malloc(sz_distances);
	assert(h_distances);

	for (uint32_t i = 0; i < n_nodes; i++) {
		h_distances[i] = UINT_MAX;
	}
	h_distances[source] = 0;

	// malloc and copy of the distances array
	cudaSafeCall(cudaMalloc((void **)&d_distances, sz_distances));
	cudaSafeCall(cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice));

	// malloc and copy of the graph
	cudaSafeCall(cudaMalloc((void **)&d_start_indices, sz_distances));
	cudaSafeCall(cudaMemcpy(d_start_indices, h_graph->start_indices, sz_distances, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void **)&d_n_neighbors, sz_distances));
	cudaSafeCall(cudaMemcpy(d_n_neighbors, h_graph->n_neighbors, sz_distances, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void **)&d_neighbors, sz_neighbors));
	cudaSafeCall(cudaMemcpy(d_neighbors, h_graph->neighbors, sz_neighbors, cudaMemcpyHostToDevice));
	cudaSafeCall(cudaMalloc((void **)&d_weights, sz_neighbors));
	cudaSafeCall(cudaMemcpy(d_weights, h_graph->weights, sz_neighbors, cudaMemcpyHostToDevice));

	fprintf(stderr, "\n");

	// Computation
	for (uint32_t i = 0; i < n_nodes - 1; i++) {
		if (i % 1000 == 0)
			fprintf(stderr, "%u / %u iterations completed\n", i, n_nodes - 1);
		cuda_bellman_ford<<<1, BLKDIM>>>(n_nodes, d_start_indices, d_n_neighbors, d_neighbors, d_weights, d_distances);
		cudaCheckError();
	}

	// copy-back of the result
	cudaSafeCall(cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost));

	// deallocation
	cudaFree(d_start_indices);
	cudaFree(d_n_neighbors);
	cudaFree(d_neighbors);
	cudaFree(d_weights);
	cudaFree(d_distances);

	return h_distances;
}

int main(void) {
	Node *list_of_nodes;
	Graph_soa *graph;
	uint32_t nodes, edges;
	uint32_t *result;

	clock_t program_start, program_end, compute_start, compute_end;

	program_start = clock();

	fprintf(stderr, "Reading input graph...");
	list_of_nodes = read_graph_adj_list(&nodes, &edges);
	fprintf(stderr, "done\n");

	graph = convert_to_soa(list_of_nodes, nodes, edges);
	destroy_graph(nodes, list_of_nodes);

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage((2 * nodes + 2 * edges) * sizeof(uint32_t));

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

	float total_seconds = (float)(program_end - program_start) / (float)CLOCKS_PER_SEC;
	float compute_seconds = (float)(compute_end - compute_start) / (float)CLOCKS_PER_SEC;

	fprintf(stderr, "\nTotal execution time: %.3f seconds\n", total_seconds);
	fprintf(stderr, "Actual execution time: %.3f seconds\n", compute_seconds);

	uint64_t total_work = (uint64_t)nodes * (uint64_t)edges;
	double throughput = (double)total_work / (double)compute_seconds;
	fprintf(stderr, "\nThroughput: %.3e relax/second\n", throughput);

	return EXIT_SUCCESS;
}