/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF2-AoS-NoSh:
	- the input graph is stored as an adjacency list (Array of Structures),
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
	CUDA kernel of Bellman-Ford's algorithm.
	A single block of |BLKDIM| threads executes a relax on each incoming arc
	of each node.
*/
__global__ void cuda_bellman_ford(uint32_t n_nodes, Node *graph, uint32_t *distances) {
	const uint32_t node = blockIdx.x;

	for (uint32_t idx = threadIdx.x; idx < graph[node].n_neighbors; idx += BLKDIM) {
		// relax the edge (u,v)
		const uint32_t u = graph[node].neighbors[idx];
		const uint32_t v = node;
		// overflow-safe check
		if (distances[v] > distances[u] && distances[v] - distances[u] > graph[node].weights[idx]) {
			distances[v] = distances[u] + graph[node].weights[idx];
		}
	}
}

/*
	This kernel is supposed to be executed from a single thread
	in a single block.
	Connects pointers to create an adjacency list on the device.
*/
__global__ void connect_pointers(Node *node, uint32_t *new_neighbors, uint32_t *new_weights) {
	const uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx != 0) {
		return;
	}

	node->neighbors = new_neighbors;
	node->weights = new_weights;
}

/*
	Executes the Bellman-Ford's algorithm on the graph |h_graph|.
	Returns a pointer to an array with |n_nodes| elements:
	each element of index |i| contains the shortest path distance from node
	|source| to node |i|.
*/
uint32_t *bellman_ford(Node *h_graph, uint32_t n_nodes, uint32_t source) {
	if (h_graph == NULL) {
		return NULL;
	}

	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
		exit(EXIT_FAILURE);
	}

	size_t sz_distances = n_nodes * sizeof(uint32_t);
	size_t sz_graph = n_nodes * sizeof(Node);

	Node *d_graph;

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
	cudaSafeCall(cudaMalloc((void **)&d_graph, sz_graph));
	cudaSafeCall(cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice));

	// copying each node's neighbors
	for (uint32_t i = 0; i < n_nodes; i++) {
		uint32_t *d_tmp_neighbors;
		uint32_t *d_tmp_weights;
		const uint32_t sz = h_graph[i].n_neighbors * sizeof(uint32_t);
		cudaSafeCall(cudaMalloc((void **)&d_tmp_neighbors, sz));
		cudaSafeCall(cudaMemcpy(d_tmp_neighbors, h_graph[i].neighbors, sz, cudaMemcpyHostToDevice));
		cudaSafeCall(cudaMalloc((void **)&d_tmp_weights, sz));
		cudaSafeCall(cudaMemcpy(d_tmp_weights, h_graph[i].weights, sz, cudaMemcpyHostToDevice));
		connect_pointers<<<1, 1>>>(&d_graph[i], d_tmp_neighbors, d_tmp_weights);
	}
	cudaCheckError();

	fprintf(stderr, "\n");

	// Computation
	for (uint32_t i = 0; i < n_nodes - 1; i++) {
		if (i % 1000 == 0)
			fprintf(stderr, "%u / %u iterations completed\n", i, n_nodes - 1);
		cuda_bellman_ford<<<n_nodes, BLKDIM>>>(n_nodes, d_graph, d_distances);
		cudaCheckError();
	}

	// copy-back of the result
	cudaSafeCall(cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost));

	// deallocation
	cudaFree(d_graph);
	cudaFree(d_distances);

	return h_distances;
}

int main(int argc, char *argv[]) {
	if (argc < 2 || argc > 3) {
		fprintf(stderr, "Usage: %s <input_file> [<solution_file>]\n\n", argv[0]);
		return EXIT_FAILURE;
	}

	Node *graph;
	uint32_t nodes;
	uint32_t edges;
	uint32_t *result;

	clock_t compute_start;
	clock_t compute_end;

	fprintf(stderr, "Reading input graph...");
	graph = read_graph_adj_list(argv[1], &nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage(nodes * sizeof(Node) + edges * 2 * sizeof(uint32_t));

	fprintf(stderr, "Computing Bellman-Ford...");
	compute_start = clock();
	result = bellman_ford(graph, nodes, 0);
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

	destroy_graph(nodes, graph);
	free(result);

	return EXIT_SUCCESS;
}