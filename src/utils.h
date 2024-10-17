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

typedef struct {
	// The index of the source node of the edge
	uint32_t start_node;

	// The index of the destination node of the edge
	uint32_t end_node;

	// The weight assigned to the edge
	float weight;
} Edge;

typedef struct {
	// Number of neighbors
	uint32_t n_neighbors;

	// Array of indices of neighbor nodes
	uint32_t *neighbors;

	// Weights of outgoing arcs to neighbors
	float *weights;
} Node;

typedef struct {
	// The index of the source node of the edge
	uint32_t *start_nodes;

	// The index of the destination node of the edge
	uint32_t *end_nodes;

	// The weight assigned to the edge
	float *weights;
} Graph;

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
void dump_solution_float(const uint32_t n_nodes, const uint32_t source, const float *const dist) {
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
void print_ram_usage(const uint32_t nbytes) {
	const double ram_usage = (double)nbytes;
	if (ram_usage < 1000.0) {
		fprintf(stderr, " %.3f bytes of RAM used\n\n", ram_usage);
	} else if (ram_usage < 1000000.0) {
		fprintf(stderr, " %.3f KBytes of RAM used\n\n", ram_usage / 1000.0);
	} else {
		fprintf(stderr, " %.3f MBytes of RAM used\n\n", ram_usage / 1000000.0);
	}
}

#endif	// UTILS_H