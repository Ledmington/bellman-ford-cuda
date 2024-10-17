/*
	CUDA implementation of the Bellman-Ford's algorithm.

	Version BF0-none-AoS-Sh:
	- the input graph is stored as an array of weighted arcs (Array of Structures),
	- the parallelization is done on the "inner cycle",
	- no mutexes
	- a shared memory buffer is used
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

#include "utils.h"

// CUDA block's size for monodimensional grid
#define BLKDIM 1024

/*
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination
   node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to an array of |n_edges| structures of type
   Edge.
*/
Edge *read_graph(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is
	   useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Edge *graph = (Edge *)malloc((*n_edges) * sizeof(Edge));
	assert(graph);

	for (uint32_t i = 0; i < *n_edges; i++) {
		float tmp;
		scanf("%u %u %f", &graph[i].start_node, &graph[i].end_node, &tmp);
		graph[i].weight = (uint32_t)tmp;

		if (graph[i].start_node >= *n_nodes || graph[i].end_node >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index\n\n", i + 1);
			exit(EXIT_FAILURE);
		}
	}

	return graph;
}

/*
	CUDA kernel of Bellman-Ford's algorithm.
	Each thread executes a relax on a single edge in each kernel call.
*/
__global__ void cuda_bellman_ford(uint32_t n_edges, Edge *graph, uint32_t *distances) {
	__shared__ Edge buffer[BLKDIM];
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
		// overflow-safe check
		if (distances[v] > distances[u] && distances[v] - distances[u] > buffer[l_idx].weight) {
			distances[v] = distances[u] + buffer[l_idx].weight;
		}
	}
}

/*
	Executes the Bellman-Ford's algorithm on the graph |h_graph|.
	Returns a pointer to an array with |n_nodes| elements:
	each element of index |i| contains the shortest path distance from node
	|source| to node |i|.
*/
uint32_t *bellman_ford(Edge *h_graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (h_graph == NULL)
		return NULL;
	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist\n\n", source);
		exit(EXIT_FAILURE);
	}

	size_t sz_distances = n_nodes * sizeof(uint32_t);
	size_t sz_graph = n_edges * sizeof(Edge);

	Edge *d_graph;

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

int main(void) {
	Edge *graph;
	uint32_t nodes, edges;
	uint32_t *result;

	clock_t program_start, program_end, compute_start, compute_end;

	program_start = clock();

	fprintf(stderr, "Reading input graph...");
	graph = read_graph(&nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage(sizeof(Edge) * edges);

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

	float total_seconds = (float)(program_end - program_start) / (float)CLOCKS_PER_SEC;
	float compute_seconds = (float)(compute_end - compute_start) / (float)CLOCKS_PER_SEC;

	fprintf(stderr, "\nTotal execution time: %.3f seconds\n", total_seconds);
	fprintf(stderr, "Actual execution time: %.3f seconds\n", compute_seconds);

	unsigned long long total_work = (unsigned long long)nodes * (unsigned long long)edges;
	double throughput = (double)total_work / (double)compute_seconds;
	fprintf(stderr, "\nThroughput: %.3e relax/second\n", throughput);

	return EXIT_SUCCESS;
}