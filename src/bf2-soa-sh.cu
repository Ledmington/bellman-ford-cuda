/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF2-SoA-Sh:
	- the input graph is stored as an adjacency list (Structure of Arrays),
	- the parallelization is done on the "inner cycle",
	- a shared memory buffer is used
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

/*
	CUDA kernel of Bellman-Ford's algorithm.
	Each block of |BLKDIM| threads executes a relax on each incoming edge
	of one node.
*/
__global__ void cuda_bellman_ford(uint32_t n_nodes, uint32_t *start_indices, uint32_t *n_neighbors, uint32_t *neighbors,
								  uint32_t *weights, uint32_t *distances) {
	__shared__ uint32_t sh_buffer[2 * BLKDIM];
	const uint32_t node = blockIdx.x;
	const uint32_t l_idx = 2 * threadIdx.x;

	for (uint32_t g_idx = threadIdx.x; g_idx < n_neighbors[node]; g_idx += BLKDIM) {
		sh_buffer[l_idx] = neighbors[start_indices[node] + g_idx];
		sh_buffer[l_idx + 1] = weights[start_indices[node] + g_idx];

		// relax the edge (u,v)
		const uint32_t u = sh_buffer[l_idx];
		const uint32_t v = node;
		// overflow-safe check
		if (distances[v] > distances[u] && distances[v] - distances[u] > sh_buffer[l_idx + 1]) {
			distances[v] = distances[u] + sh_buffer[l_idx + 1];
		}
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
		cuda_bellman_ford<<<n_nodes, BLKDIM>>>(n_nodes, d_start_indices, d_n_neighbors, d_neighbors, d_weights,
											   d_distances);
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

int main(int argc, char *argv[]) {
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "Usage: %s <input_file> [<solution_file>]\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	Node *list_of_nodes;
	Graph_soa *graph;
	uint32_t nodes;
	uint32_t edges;
	uint32_t *result;

	clock_t compute_start;
	clock_t compute_end;

	fprintf(stderr, "Reading input graph...");
	list_of_nodes = read_graph_adj_list(argv[1], &nodes, &edges);
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

	const float compute_seconds = (float)(compute_end - compute_start) / (float)CLOCKS_PER_SEC;
	fprintf(stderr, "\nActual execution time: %.3f seconds\n", compute_seconds);

	uint64_t total_work = (uint64_t)nodes * (uint64_t)edges;
	double throughput = (double)total_work / (double)compute_seconds;
	fprintf(stderr, "\nThroughput: %.3e relax/second\n\n", throughput);

	if (argc == 3) {
		uint32_t *distances = (uint32_t *)malloc(nodes * sizeof(uint32_t));

		fprintf(stderr, "Reading solution...");
		read_solution(argv[2], distances);
		fprintf(stderr, "done\n");

		check_solution(nodes, distances, result);

		free(distances);
	} else {
		fprintf(stderr, "Dumping solution...");
		dump_solution(nodes, 0, result);
		fprintf(stderr, "done\n");
	}

	free(graph->start_indices);
	free(graph->n_neighbors);
	free(graph->neighbors);
	free(graph->weights);
	free(graph);
	free(result);

	return EXIT_SUCCESS;
}