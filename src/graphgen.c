/*
	graphgen.c - Random graph generator

	Generates an undirected unweighted random graph based on the Erdos-Renyi model.

	The output is formatted as follows:
	the first line contains three numbers: |n_nodes| |n_arcs| 1
	the following |n_arcs| lines contain:
	|source_node_index| |destination_node_index| 1.0

	Node indexes are 0-based.

	To compile:
	gcc -std=c99 -Wall -Wpedantic graphgen.c -o graphgen

	To run:
	./graphgen N p [min-weight max-weight] > graph.txt

	N is the number of nodes in the graph,
	p is the probability to generate an arc.

	If specified, each arc will have a weight in [min_weight; max_weight).
	Otherwise, each arc is generated with weight 1.
*/

#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

double randab(double a, double b) { return (double)rand() / (double)RAND_MAX * (b - a) + a; }

int main(int argc, char* argv[]) {
	srand(time(NULL));

	if (argc != 3 && argc != 5) {
		fprintf(stderr, "Usage: \"%s N p [min_weight max_weight]\"\n", argv[0]);
		fprintf(stderr, " N is the number of nodes in the graph\n");
		fprintf(stderr, " p is the probability to generate an arc\n");
		fprintf(stderr,
				" If specified, each arc will have a weight in [min_weight; max_weight).\n"
				" Otherwise, each arc is generated with weight 1.\n");
		return EXIT_FAILURE;
	}

	uint32_t N = strtoul(argv[1], NULL, 10);
	double p = strtod(argv[2], NULL);
	double min_weight = 1.0;
	double max_weight = 1.0;

	if (p < 0.0 || p > 1.0) {
		fprintf(stderr, "ERROR: invalid value of p\n");
		exit(EXIT_FAILURE);
	}

	if (argc == 5) {
		min_weight = strtod(argv[3], NULL);
		max_weight = strtod(argv[4], NULL);
	}

	if (min_weight < 0 || max_weight < 0 || min_weight > max_weight) {
		fprintf(stderr, "ERROR: invalid values of min_weight and max_weight\n");
		exit(EXIT_FAILURE);
	}

	fprintf(stderr, "N: %u\np: %f\n", N, p);
	if (argc == 5) {
		fprintf(stderr, "Weights in [%f; %f)\n", min_weight, max_weight);
	}

	uint32_t n_edges = 0;
	double** g;

	g = (double**)malloc(N * sizeof(double*));
	assert(g);
	for (uint32_t i = 0; i < N; i++) {
		g[i] = (double*)malloc(N * sizeof(double));
		assert(g[i]);
	}

	for (uint32_t i = 0; i < N; i++) {
		for (uint32_t j = 0; j < N; j++) {
			if (i == j) {
				g[i][j] = HUGE_VAL;
			} else {
				if (randab(0.0, 1.0) <= p) {
					g[i][j] = randab(min_weight, max_weight);
					n_edges++;
				} else {
					g[i][j] = HUGE_VAL;
				}
			}
		}
	}

	fprintf(stderr, "n. arcs (created / total): %u / %u\n", n_edges, N * (N - 1));
	fprintf(stderr, "density: %.6f\n", (float)n_edges / (float)(N * (N - 1)));

	printf("%u %u 1\n", N, n_edges);
	for (uint32_t i = 0; i < N; i++) {
		for (uint32_t j = 0; j < N; j++) {
			if (!isinf(g[i][j])) {
				printf("%u %u %f\n", i, j, g[i][j]);
			}
		}
	}

	for (uint32_t i = 0; i < N; i++) {
		free(g[i]);
	}
	free(g);

	return EXIT_SUCCESS;
}
