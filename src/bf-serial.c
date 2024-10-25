/*
	Serial CPU implementation of the Bellman-Ford's algorithm.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <time.h>
#include <float.h>

#include "utils.h"

/*
	Executes the Bellman-Ford's algorithm on the graph |h_graph|.
	Returns a pointer to an array with |n_nodes| elements:
	each element of index |i| contains the shortest path distance from node
	|source| to node |i|.
*/
float *bellman_ford(Edge *graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (graph == NULL) {
		return NULL;
	}

	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist.\n\n", source);
		exit(EXIT_FAILURE);
	}

	float *D = (float *)malloc(n_nodes * sizeof(float));
	assert(D);

	for (uint32_t i = 0; i < n_nodes; i++) {
		D[i] = FLT_MAX;
	}
	D[source] = 0;

	for (uint32_t i = 0; i < n_nodes - 1; i++) {
		if (i % 1000 == 0) {
			fprintf(stderr, "%u / %u completed iterations\n", i, n_nodes - 1);
		}

		for (uint32_t e = 0; e < n_edges; e++) {
			const uint32_t u = graph[e].start_node;
			const uint32_t v = graph[e].end_node;
			// overflow-safe check
			if (D[v] > D[u] && D[v] - D[u] > graph[e].weight) {
				D[v] = D[u] + graph[e].weight;
			}
		}
	}

	return D;
}

int main(void) {
	Edge *graph;
	uint32_t nodes, edges;
	float *result;

	clock_t program_start, program_end, compute_start, compute_end;

	program_start = clock();

	fprintf(stderr, "Reading input graph...");
	graph = read_graph(&nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	print_ram_usage(sizeof(Edge) * edges);

	fprintf(stderr, "Computing Bellman-Ford...\n");
	compute_start = clock();
	result = bellman_ford(graph, nodes, edges, 0);
	compute_end = clock();
	fprintf(stderr, "done\n\n");

	fprintf(stderr, "Dumping solution...");
	dump_solution_f(nodes, 0, result);
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