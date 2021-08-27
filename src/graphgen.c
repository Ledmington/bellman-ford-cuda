/*
    Random graph generator
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
    graphgen.c

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
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <assert.h>
#include <math.h>

double randab(double a, double b) {
    return (double)rand() / (double)RAND_MAX * (b-a) + a;
}

int main ( int argc, char** argv ) {
    srand(time(NULL));

    if(argc != 3 && argc != 5) {
        fprintf(stderr, "Usage: \"%s N p [min_weight max_weight]\"\n", argv[0]);
        fprintf(stderr, " N is the number of nodes in the graph\n");
        fprintf(stderr, " p is the probability to generate an arc\n");
        fprintf(stderr, " If specified, each arc will have a weight in [min_weight; max_weight).\n"
                        " Otherwise, each arc is generated with weight 1.\n");
        return EXIT_FAILURE;
    }

    unsigned int N = atoi(argv[1]);
    double p = strtod(argv[2], NULL);
    double min_weight = 1.0f;
    double max_weight = 1.0f;

    if(p < 0.0f || p > 1.0f) {
        fprintf(stderr, "ERROR: invalid value of p\n");
        exit(EXIT_FAILURE);
    }

    if(argc == 5) {
        min_weight = strtod(argv[3], NULL);
        max_weight = strtod(argv[4], NULL);
    }

    if(min_weight < 0 || max_weight < 0 || min_weight > max_weight) {
        fprintf(stderr, "ERROR: invalid values of min_weight and max_weight\n");
        exit(EXIT_FAILURE);
    }

    fprintf(stderr, "N: %u\np: %f\n", N, p);
    if(argc == 5) {
        fprintf(stderr, "Weights in [%f; %f)\n", min_weight, max_weight);
    }

    unsigned int n_archi = 0;
    double **g;

    g = (double**) malloc(N * sizeof(double*));
    assert(g);
    for(unsigned int i=0; i<N; i++) {
        g[i] = (double*) malloc(N * sizeof(double));
        assert(g[i]);
    }

    for(unsigned int i=0; i<N; i++) {
        for(unsigned int j=0; j<N; j++) {
            if(i==j) g[i][j] = HUGE_VAL;
            else {
                if(randab(0.0, 1.0) <= p) {
                    g[i][j] = randab(min_weight, max_weight);
                    n_archi++;
                }
                else {
                    g[i][j] = HUGE_VAL;
                }
            }
        }
    }

    fprintf(stderr, "n. arcs (created / total): %u / %u\n", n_archi, N*(N-1));
    fprintf(stderr, "density: %.6f\n", (float)n_archi/(float)(N*(N-1)));

    printf("%u %u 1\n", N, n_archi);
    for(unsigned int i=0; i<N; i++) {
        for(unsigned int j=0; j<N; j++) {
            if(!isinf(g[i][j])) {
                printf("%u %u %f\n", i, j, g[i][j]);
            }
        }
    }

    for(unsigned int i=0; i<N; i++) {
        free(g[i]);
    }
    free(g);

    return EXIT_SUCCESS;
}