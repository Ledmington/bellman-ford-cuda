/*
    Implementazione CUDA dell'algoritmo di Bellman-Ford.
    Versione BF0-min:
    - il grafo è memorizzato come una lista di archi pesati,
    - la parallelizzazione è effettuata su ciclo che itera gli archi
         (il "ciclo interno"),
    - si utilizza atomicMin per l'aggiornamento atomico

    Per compilare:
    nvcc -arch=<cuda_capability> bf0-min.cu -o bf0-min

    Per eseguire:
    ./bf0-min < test/graph.txt > solution.txt
*/

#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// La dimensione del blocco CUDA monodimensionale
#define BLKDIM 1024

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
    Stampa la soluzione di Bellman-Ford su stdin.

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
    Kernel CUDA per l'algoritmo di Bellman-Ford.
    Ogni thread esegue V-1 rilassamenti su un nodo del grafo.
*/
__global__ void cuda_bellman_ford (unsigned int n_nodes,
                                   unsigned int n_edges,
                                   Edge* graph,
                                   unsigned int *distances) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_edges) {
        // relax the edge (u,v)
        const unsigned int u = graph[idx].start_node;
        const unsigned int v = graph[idx].end_node;
        atomicMin(&distances[v], distances[u] + graph[idx].weight);
    }
}

/*
    Esegue l'algoritmo di Bellman-Ford sul grafo passato in input.
    Restituisce un puntatore ad un vettore con |n_nodes| elementi:
    ciascuno elemento di indice i contiene la distanza del cammino minimo
    dal nodo |source| al nodo i.
*/
unsigned int* bellman_ford ( Edge* h_graph, unsigned int n_nodes, unsigned int n_edges, unsigned int source ) {
    if(h_graph == NULL) return NULL;
    if(source >= n_nodes) {
        fprintf(stderr, "ERRORE: il nodo sorgente %u non esiste\n\n", source);
        exit(EXIT_FAILURE);
    }

    size_t sz_distances = n_nodes * sizeof(unsigned int);
    size_t sz_graph = n_edges * sizeof(Edge);

    Edge* d_graph;

    unsigned int *d_distances;
    unsigned int *h_distances = (unsigned int*) malloc(sz_distances);
    assert(h_distances);

    for(unsigned int i=0; i<n_nodes; i++) {
        // TODO: change this to a valid infinity
        h_distances[i] = 0x00ffffff; // this doesn't overflow
        //h_distances[i] = 0xffffffff; // this creates overflow
    }
    h_distances[source] = 0;

    // malloc and copy of the distances array
    cudaSafeCall( cudaMalloc((void**)&d_distances, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice) );

    // malloc and copy of the graph
    cudaSafeCall( cudaMalloc((void**)&d_graph, sz_graph) );
    cudaSafeCall( cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice) );

    for(unsigned int i=0; i<n_nodes-1; i++) {
        // kernel invocation
        cuda_bellman_ford <<< (n_edges+BLKDIM-1) / BLKDIM, BLKDIM >>>(n_nodes, n_edges, d_graph, d_distances);
        cudaCheckError();
    }

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_graph);
    cudaFree(d_distances);

    return h_distances;
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

    fprintf(stderr, "Dumping solution...");
    dump_solution(nodes, 0, result);
    fprintf(stderr, "done\n");

    free(graph);
    free(result);

    return EXIT_SUCCESS;
}