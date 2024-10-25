/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF0-mutex-AoS-Sh:
	- the input graph is stored as an array of weighted arcs (Array of Structures),
	- the parallelization is done on the "inner cycle",
	- an atomic operation is used for the update of distances
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
	Each thread executes a relax on a single edge in each kernel call.
*/
__global__ void cuda_bellman_ford(uint32_t n_edges, Edge_f *graph, float *distances) {
	union {
		float vf;
		int vi;
	} oldval, newval;

	__shared__ Edge_f buffer[BLKDIM];
	uint32_t g_idx = blockIdx.x * blockDim.x + threadIdx.x;
	uint32_t l_idx = threadIdx.x;

	// Filling the shared memory buffer
	if (g_idx < n_edges) {
		buffer[l_idx] = graph[g_idx];
	}
	__syncthreads();

	if (g_idx < n_edges) {
		// relax the edge (u,v)
		const uint32_t u = buffer[l_idx].start_node;
		const uint32_t v = buffer[l_idx].end_node;

		if (distances[u] + buffer[l_idx].weight < distances[v]) {
			do {
				oldval.vf = distances[v];
				newval.vf = distances[u] + buffer[l_idx].weight;
				newval.vf = fminf(oldval.vf, newval.vf);
			} while (atomicCAS((int *)&distances[v], oldval.vi, newval.vi) != newval.vi);
		}
	}
}

/*
	Executes the Bellman-Ford's algorithm on the graph |h_graph|.
	Returns a pointer to an array with |n_nodes| elements:
	each element of index |i| contains the shortest path distance from node
	|source| to node |i|.
*/
float *bellman_ford(Edge_f *h_graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (h_graph == NULL) {
		return NULL;
	}

	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
		exit(EXIT_FAILURE);
	}

	const size_t sz_distances = n_nodes * sizeof(float);
	const size_t sz_graph = n_edges * sizeof(Edge_f);

	Edge_f *d_graph;

	float *d_distances;
	float *h_distances = (float *)malloc(sz_distances);
	assert(h_distances);

	for (uint32_t i = 0; i < n_nodes; i++) {
		h_distances[i] = HUGE_VAL;
	}
	h_distances[source] = 0.0f;

	// malloc and copy of the distances array
	cudaSafeCall(cudaMalloc((void **)&d_distances, sz_distances));
	cudaSafeCall(cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice));

	// malloc and copy of the graph
	cudaSafeCall(cudaMalloc((void **)&d_graph, sz_graph));
	cudaSafeCall(cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice));

	for (uint32_t i = 0; i < n_nodes - 1; i++) {
		cuda_bellman_ford<<<(n_edges + BLKDIM - 1) / BLKDIM, BLKDIM>>>(n_edges, d_graph, d_distances);
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

	Edge_f *graph;
	uint32_t nodes;
	uint32_t edges;
	float *result;

	clock_t compute_start;
	clock_t compute_end;

	fprintf(stderr, "Reading input graph...");
	graph = read_graph_f(argv[1], &nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage(sizeof(Edge_f) * edges);

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
		float *distances = (float *)malloc(nodes * sizeof(float));

		fprintf(stderr, "Reading solution...");
		read_solution_f(argv[2], distances);
		fprintf(stderr, "done\n");

		check_solution_f(nodes, distances, result);

		free(distances);
	} else {
		fprintf(stderr, "Dumping solution...");
		dump_solution_f(nodes, 0, result);
		fprintf(stderr, "done\n");
	}

	free(graph);
	free(result);

	return EXIT_SUCCESS;
}