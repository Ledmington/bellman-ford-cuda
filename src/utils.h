/*
	utils.h - Some utility functions for Bellman-Ford's CUDA implementation
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

#ifndef UTILS_H
#define UTILS_H

#include <stdio.h>
#include <stdint.h>
#include <math.h>
#include <limits.h>

/*
	All algorithms use edges with integer weights.
 */
typedef struct {
	// The index of the source node of the edge
	uint32_t start_node;

	// The index of the destination node of the edge
	uint32_t end_node;

	// The weight assigned to the edge
	uint32_t weight;
} Edge;

/*
	Only the algorithms of the group 'bf0-mutex' use edges with float weights.
 */
typedef struct {
	// The index of the source node of the edge
	uint32_t start_node;

	// The index of the destination node of the edge
	uint32_t end_node;

	// The weight assigned to the edge
	float weight;
} Edge_f;

typedef struct {
	// Number of neighbors
	uint32_t n_neighbors;

	// Array of indices of neighbor nodes
	uint32_t *neighbors;

	// Weights of outgoing arcs to neighbors
	uint32_t *weights;
} Node;

typedef struct {
	// The index of the source node of the edge
	uint32_t *start_nodes;

	// The index of the destination node of the edge
	uint32_t *end_nodes;

	// The weight assigned to the edge
	uint32_t *weights;
} Graph;

typedef struct {
	// The index of the source node of the edge
	uint32_t *start_nodes;

	// The index of the destination node of the edge
	uint32_t *end_nodes;

	// The weight assigned to the edge
	float *weights;
} Graph_f;

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
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to an array of |n_edges| structures of type Edge_f.
*/
Edge_f *read_graph_f(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Edge_f *graph = (Edge_f *)malloc(*n_edges * sizeof(Edge_f));
	assert(graph);

	for (uint32_t i = 0; i < *n_edges; i++) {
		float tmp;
		scanf("%u %u %f", &graph[i].start_node, &graph[i].end_node, &tmp);
		graph[i].weight = tmp;

		if (graph[i].start_node >= *n_nodes || graph[i].end_node >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index.\n\n", i + 1);
			exit(EXIT_FAILURE);
		}
	}

	return graph;
}

/*
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination
   node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to an array of |n_nodes| structures of type Node.
*/
Node *read_graph_adj_list(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is
	   useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Node *graph = (Node *)malloc((*n_nodes) * sizeof(Node));
	assert(graph);

	for (uint32_t i = 0; i < *n_nodes; i++) {
		graph[i].n_neighbors = 0;
		graph[i].neighbors = NULL;
		graph[i].weights = NULL;
	}

	for (uint32_t i = 0; i < *n_edges; i++) {
		uint32_t start_node, end_node;
		float weight;
		scanf("%u %u %f", &start_node, &end_node, &weight);

		if (start_node >= *n_nodes || end_node >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index\n\n", i + 1);
			exit(EXIT_FAILURE);
		}

		graph[start_node].neighbors =
			(uint32_t *)realloc(graph[start_node].neighbors, (graph[start_node].n_neighbors + 1) * sizeof(uint32_t *));
		assert(graph[start_node].neighbors);
		graph[start_node].weights =
			(uint32_t *)realloc(graph[start_node].weights, (graph[start_node].n_neighbors + 1) * sizeof(uint32_t *));
		assert(graph[start_node].weights);
		graph[start_node].neighbors[graph[start_node].n_neighbors] = end_node;
		graph[start_node].weights[graph[start_node].n_neighbors] = (uint32_t)weight;
		graph[start_node].n_neighbors++;
	}

	return graph;
}

/*
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to a Graph structure.
*/
Graph *read_graph_soa(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Graph *graph = (Graph *)malloc(sizeof(Graph));
	assert(graph);

	graph->start_nodes = (uint32_t *)malloc((*n_edges) * sizeof(uint32_t));
	assert(graph->start_nodes);
	graph->end_nodes = (uint32_t *)malloc((*n_edges) * sizeof(uint32_t));
	assert(graph->end_nodes);
	graph->weights = (uint32_t *)malloc((*n_edges) * sizeof(uint32_t));
	assert(graph->weights);

	for (uint32_t i = 0; i < *n_edges; i++) {
		float tmp;
		scanf("%u %u %f", &graph->start_nodes[i], &graph->end_nodes[i], &tmp);
		graph->weights[i] = (uint32_t)tmp;

		if (graph->start_nodes[i] >= *n_nodes || graph->end_nodes[i] >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index.\n\n", i + 1);
			exit(EXIT_FAILURE);
		}
	}

	return graph;
}

/*
	Reads a graph from stdin formatted as follows:
	first line: |number of nodes| |number of arcs| n
	each one of the other |number of arcs| lines: |source node| |destination node| |arc weight|

	The variables pointed by |n_nodes| and |n_edges| are modified accordingly.

	This function returns a pointer to a Graph_f structure.
*/
Graph_f *read_graph_soa_f(uint32_t *n_nodes, uint32_t *n_edges) {
	/*
		|tmp| is necessary to read the third value of the first line, which is useless
	*/
	uint32_t tmp;
	scanf("%u %u %u", n_nodes, n_edges, &tmp);

	Graph_f *graph = (Graph_f *)malloc(sizeof(Graph_f));
	assert(graph);

	graph->start_nodes = (uint32_t *)malloc((*n_edges) * sizeof(uint32_t));
	assert(graph->start_nodes);
	graph->end_nodes = (uint32_t *)malloc((*n_edges) * sizeof(uint32_t));
	assert(graph->end_nodes);
	graph->weights = (float *)malloc((*n_edges) * sizeof(float));
	assert(graph->weights);

	for (uint32_t i = 0; i < *n_edges; i++) {
		scanf("%u %u %f", &graph->start_nodes[i], &graph->end_nodes[i], &graph->weights[i]);

		if (graph->start_nodes[i] >= *n_nodes || graph->end_nodes[i] >= *n_nodes) {
			fprintf(stderr, "ERROR at line %u: invalid node index.\n\n", i + 1);
			exit(EXIT_FAILURE);
		}
	}

	return graph;
}

void destroy_graph(uint32_t nodes, Node *graph) {
	for (uint32_t i = 0; i < nodes; i++) {
		free(graph[i].neighbors);
		free(graph[i].weights);
	}
	free(graph);
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
void dump_solution(const uint32_t n_nodes, const uint32_t source, const uint32_t *const dist) {
	printf("%u\n%u\n", n_nodes, source);

	for (uint32_t i = 0; i < n_nodes; i++) {
		printf("%u %u\n", i, dist[i]);
	}
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
void dump_solution_f(const uint32_t n_nodes, const uint32_t source, const float *const dist) {
	printf("%u\n%u\n", n_nodes, source);

	for (uint32_t i = 0; i < n_nodes; i++) {
		printf("%u", i);
		if (isinf(dist[i])) {
			printf(" %u\n", UINT_MAX);
		} else {
			printf(" %u\n", (uint32_t)dist[i]);
		}
	}
}

/*
	Prints correctly the amount of RAM used.
*/
void print_ram_usage(const uint64_t nbytes) {
	const double ram_usage = (double)nbytes;
	if (ram_usage < 1000.0) {
		fprintf(stderr, " %.3f bytes of RAM used\n\n", ram_usage);
	} else if (ram_usage < 1000000.0) {
		fprintf(stderr, " %.3f KBytes of RAM used\n\n", ram_usage / 1000.0);
	} else {
		fprintf(stderr, " %.3f MBytes of RAM used\n\n", ram_usage / 1000000.0);
	}
}

#ifdef __CUDACC__

#include <stdio.h>
#include <stdlib.h>

/* from https://gist.github.com/ashwin/2652488 */

#define cudaSafeCall(err) __cudaSafeCall(err, __FILE__, __LINE__)
#define cudaCheckError() __cudaCheckError(__FILE__, __LINE__)

inline void __cudaSafeCall(cudaError err, const char *file, const int line) {
#ifndef NO_CUDA_CHECK_ERROR
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaSafeCall() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
#endif
}

inline void __cudaCheckError(const char *file, const int line) {
#ifndef NO_CUDA_CHECK_ERROR
	cudaError err = cudaGetLastError();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}

	/* More careful checking. However, this will affect performance.
	   Comment away if needed. */
	err = cudaDeviceSynchronize();
	if (cudaSuccess != err) {
		fprintf(stderr, "cudaCheckError() with sync failed at %s:%i : %s\n", file, line, cudaGetErrorString(err));
		abort();
	}
#endif
}

#endif	// __CUDACC__

#endif	// UTILS_H
