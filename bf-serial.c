/*
    Implementazione seriale su CPU dell'algoritmo di Bellman-Ford.

    Per compilare:
    gcc -std=c99 -Wall -Wpedantic bf-serial.c -o bf-serial

    Per eseguire:
    ./bf-serial < test/graph.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

typedef struct {
    unsigned int start_node;
    unsigned int end_node;
    unsigned int weight;
} Edge;

/*
    Legge un grafo da stdin formattato come segue:
    prima riga: |numero nodi| |numero archi| n
    tutte le altre |numero archi| righe: |nodo sorgente| |nodo destinazione| |peso arco|

    Le variabili puntate da |n_nodes| e |n_edges| sono modificate opportunamente.

    Retituisce un puntatore ad un array di |n_edges| strutture Edge.
*/
Edge* read_graph ( unsigned int *n_nodes, unsigned int *n_edges ) {
    /*
        |tmp| è necessaria per leggere il terzo valore della prima riga, che però non serve
    */
    unsigned int tmp;
    scanf("%u %u %u", n_nodes, n_edges, &tmp);
    //fprintf(stderr, "Nodes: %u\n", n_nodes);
    //fprintf(stderr, "Edges: %u\n", n_edges);

    Edge *graph = (Edge*) malloc(*n_edges * sizeof(Edge));
    assert(graph);

    for(unsigned int i=0; i<*n_edges; i++) {
        float tmp;
        scanf("%u %u %f", &graph[i].start_node, &graph[i].end_node, &tmp);
        graph[i].weight = (unsigned int)tmp;

        if(graph[i].start_node >= *n_nodes || graph[i].end_node >= *n_nodes) {
            fprintf(stderr, "ERRORE alla riga %u:\nIndice del nodo non valido.\n\n", i+1);
            exit(EXIT_FAILURE);
        }
    }

    return graph;
}

/*
    Esegue l'algoritmo di Bellman-Ford sul grafo passato in input.
    Restituisce un puntatore ad un vettore con |n_nodes| elementi:
    ciascuno elemento di indice i contiene la distanza del cammino minimo
    dal nodo |source| al nodo i.
*/
unsigned int* bellman_ford ( Edge* graph, unsigned int n_nodes, unsigned int n_edges, unsigned int source ) {
    if(graph == NULL) return NULL;
    if(source >= n_nodes) {
        fprintf(stderr, "ERRORE: il nodo sorgente %u non esiste\n\n", source);
        exit(EXIT_FAILURE);
    }

    unsigned int *D = (unsigned int*) malloc(n_nodes * sizeof(unsigned int));
    assert(D);

    for(unsigned int i=0; i<n_nodes; i++) {
        D[i] = 0xffffffff;
    }
    D[source] = 0;

    for(unsigned int i=0; i<n_nodes-1; i++) {
        /*
        if(i%1000 == 0) {
            fprintf(stderr, "It. %u / %u\n", i, n_nodes-1);
        }
        */
        for(unsigned int e=0; e<n_edges; e++) {
            if(D[graph[e].start_node] + graph[e].weight < D[graph[e].end_node]) {
                D[graph[e].end_node] = D[graph[e].start_node] + graph[e].weight;
            }
        }
    }

    return D;
}

int main ( void ) {

    Edge *graph;
    unsigned int nodes, edges;
    unsigned int *result;

    fprintf(stderr, "Reading input graph...");
    graph = read_graph(&nodes, &edges);
    fprintf(stderr, "done\n");

    fprintf(stderr, "Computing Bellman-Ford...");
    result = bellman_ford(graph, nodes, edges, 0);
    fprintf(stderr, "done\n");

    fprintf(stderr, "\n");

    for(unsigned int i=0; i<10; i++) {
        fprintf(stderr, "D[%u]: %u\n", i, result[i]);
    }

    free(graph);
    free(result);

    return EXIT_SUCCESS;
}