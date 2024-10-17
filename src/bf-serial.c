/*
	Serial implementation of the Bellman-Ford's algorithm
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
	Serial CPU implementation of the Bellman-Ford's algorithm.

	To compile:
	gcc -std=c99 -Wall -Wpedantic bf-serial.c -o bf-serial

	To run:
	./bf-serial < test/graph.txt > solution.txt
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

typedef struct {
	// The index of the source node of the edge
	uint32_t start_node;

	// The index of the destination node of the edge
	uint32_t end_node;

	// The weight assigned to the edge
	uint32_t weight;
} Edge;

/*
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to an array of |n_edges| structures of type Edge.
*/
Edge *read_graph(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Edge *graph = (Edge *)malloc(*n_edges * sizeof(Edge));
	assert(graph);

	for (uint32_t i = 0; i < *n_edges; i++) {
		float tmp;
		scanf("%u %u %f", &graph[i].start_node, &graph[i].end_node, &tmp);
		graph[i].weight = (uint32_t)tmp;

		if (graph[i].start_node >= *n_nodes || graph[i].end_node >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index.\n\n", i + 1);
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
void dump_solution(uint32_t n_nodes, uint32_t source, uint32_t *dist) {
	printf("%u\n%u\n", n_nodes, source);

	for (uint32_t i = 0; i < n_nodes; i++) {
		printf("%u %u\n", i, dist[i]);
	}
}

/*
	Executes the Bellman-Ford's algorithm on the graph |h_graph|.
	Returns a pointer to an array with |n_nodes| elements:
	each element of index |i| contains the shortest path distance from node
	|source| to node |i|.
*/
uint32_t *bellman_ford(Edge *graph, uint32_t n_nodes, uint32_t n_edges, uint32_t source) {
	if (graph == NULL)
		return NULL;
	if (source >= n_nodes) {
		fprintf(stderr, "ERROR: source node %u does not exist.\n\n", source);
		exit(EXIT_FAILURE);
	}

	uint32_t *D = (uint32_t *)malloc(n_nodes * sizeof(uint32_t));
	assert(D);

	for (uint32_t i = 0; i < n_nodes; i++) {
		D[i] = UINT_MAX;
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
	uint32_t *result;

	clock_t program_start, program_end, compute_start, compute_end;

	program_start = clock();

	fprintf(stderr, "Reading input graph...");
	graph = read_graph(&nodes, &edges);
	fprintf(stderr, "done\n");

	fprintf(stderr, "\nGraph data:\n");
	fprintf(stderr, " %7u nodes\n", nodes);
	fprintf(stderr, " %7u arcs\n", edges);

	float ram_usage = (float)(sizeof(Edge) * edges);
	if (ram_usage < 1024.0f) {
		fprintf(stderr, " %.3f bytes of RAM used\n\n", ram_usage);
	} else if (ram_usage < 1024.0f * 1024.0f) {
		fprintf(stderr, " %.3f KBytes of RAM used\n\n", ram_usage / 1024.0f);
	} else {
		fprintf(stderr, " %.3f MBytes of RAM used\n\n", ram_usage / (1024.0f * 1024.0f));
	}

	fprintf(stderr, "Computing Bellman-Ford...\n");
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