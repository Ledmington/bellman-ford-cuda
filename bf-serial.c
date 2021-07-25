/*
    Implementazione seriale su CPU dell'algoritmo di Bellman-Ford.

    Per compilare:
    gcc -std=c99 -Wall -Wpedantic bf-serial.c -o bf-serial

    Per eseguire:
    ./bf-serial < test/graph.txt > solution.txt
*/

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <limits.h>
#include <time.h>

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
            fprintf(stderr, "ERRORE alla riga %u: indice del nodo non valido.\n\n", i+1);
            exit(EXIT_FAILURE);
        }
    }

    return graph;
}

/*
    Stampa la soluzione di Bellman-Ford su stdout.

    L'output è formattato come segue:

    numero_di_nodi
    nodo_sorgente
    nodo_0 distanza_al_nodo_0
    nodo_1 distanza_al_nodo_1
    nodo_2 distanza_al_nodo_2
    ...
*/
void dump_solution (unsigned int n_nodes, unsigned int source, unsigned int *dist) {
    printf("%u\n%u\n", n_nodes, source);

    for(unsigned int i=0; i<n_nodes; i++) {
        printf("%u %u\n", i, dist[i]);
    }
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
        D[i] = UINT_MAX;
    }
    D[source] = 0;

    for(unsigned int i=0; i<n_nodes-1; i++) {
        if(i%1000 == 0) {
            fprintf(stderr, "%u / %u iterazioni completate\n", i, n_nodes-1);
        }

        for(unsigned int e=0; e<n_edges; e++) {
            const unsigned int u = graph[e].start_node;
            const unsigned int v = graph[e].end_node;
            // controllo overflow-safe
            if(D[v] > D[u] && D[v]-D[u] > graph[e].weight){
                D[v] = D[u] + graph[e].weight;
            }
        }
    }

    return D;
}

int main ( void ) {

    Edge *graph;
    unsigned int nodes, edges;
    unsigned int *result;

    clock_t program_start, program_end, compute_start, compute_end;

    program_start = clock();

    fprintf(stderr, "Lettura grafo di input...");
    graph = read_graph(&nodes, &edges);
    fprintf(stderr, "OK\n");

    fprintf(stderr, "\nDati del grafo:\n");
    fprintf(stderr, "%u nodi\n", nodes);
    fprintf(stderr, "%u archi\n", edges);
    fprintf(stderr, "%f MBytes di RAM utilizzata\n\n", (float)(sizeof(Edge)*edges)/(float)(1024*1024));

    fprintf(stderr, "Esecuzione Bellman-Ford...\n");
    compute_start = clock();
    result = bellman_ford(graph, nodes, edges, 0);
    compute_end = clock();
    fprintf(stderr, "OK\n\n");

    fprintf(stderr, "Scrittura soluzione...");
    dump_solution(nodes, 0, result);
    fprintf(stderr, "OK\n");

    free(graph);
    free(result);

    program_end = clock();

    fprintf(stderr, "\nTempo totale di esecuzione: %.3f secondi\n", (float)(program_end-program_start) / (float)CLOCKS_PER_SEC);
    fprintf(stderr, "Tempo di calcolo effettivo: %.3f secondi\n", (float)(compute_end-compute_start) / (float)CLOCKS_PER_SEC);

    return EXIT_SUCCESS;
}