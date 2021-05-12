/*
    graphgen.c

    Genera un grafo indiretto non pesato casuale secondo il modello Erdos-Renyi.
    
    L'output è formattato come segue:
    la prima riga contiene un singolo numero, N, il numero di nodi del grafo,
    dopodichè viene stampata la matrice di adiacenza del grafo, N righe con
    N valori ciascuna separati da spazi, se il valore è 0 l'arco non esiste,
    1 se l'arco esiste.

    Come compilare:
    gcc -std=c99 -Wall -Wpedantic graphgen.c -o graphgen

    Come eseguire
    ./graphgen N p > graph.txt

    N indica il numero di nodi del grafo da generare,
    p è la probabilità con cui generare gli archi.
*/

#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>

double randab(double a, double b) {
    return (double)rand() / (double)RAND_MAX * (b-a) + a;
}

int main ( int argc, char** argv ) {
    srand(time(NULL));

    if(argc != 3) {
        fprintf(stderr, "Utilizzo: \"%s N p\"\n", argv[0]);
        fprintf(stderr, " N e' il numero di nodi del grafo\n");
        fprintf(stderr, " p e' la probabilita' con cui generare gli archi\n");
        return EXIT_FAILURE;
    }

    unsigned int N = atoi(argv[1]);
    double p = strtod(argv[2], NULL);

    fprintf(stderr, "N: %u\np: %f\n", N, p);
    unsigned int n_archi = 0;

    bool g[N][N];

    for(unsigned int i=0; i<N; i++) {
        g[i][i] = false;
        for(unsigned int j=i+1; j<N; j++) {
            if(randab(0.0, 1.0) <= p) {
                g[i][j] = g[j][i] = true;
                n_archi++;
            }
            else {
                g[i][j] = g[j][i] = false;
            }
        }
    }

    fprintf(stderr, "n. archi (creati / totali): %u / %u\n", n_archi, N*(N-1)/2);

    printf("%u\n", N);
    for(unsigned int i=0; i<N; i++) {
        for(unsigned int j=0; j<N; j++) {
            printf("%u", g[i][j] ? 1 : 0);
            if(j != N-1) printf(" ");
        }
        printf("\n");
    }

    return EXIT_SUCCESS;
}