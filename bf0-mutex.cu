/*
    Implementazione CUDA dell'algoritmo di Bellman-Ford.
    Versione BF0-mutex:
    - il grafo è memorizzato come una lista di archi pesati,
    - la parallelizzazione è effettuata su ciclo che itera gli archi
         (il "ciclo interno"),
    - si utilizzano mutex sotto forma di un array di unsigned int

    Per compilare:
    nvcc -arch=<cuda_capability> bf0-mutex.cu -o bf0-mutex

    Per eseguire:
    ./bf0-mutex < test/graph.txt > solution.txt
*/

#include "hpc.h"

#include <stdio.h>
#include <stdlib.h>
#include <assert.h>

// La dimensione del blocco CUDA monodimensionale
#define BLKDIM 1024

/*
    Con 1 viene invocato V-1 volte un kernel che esegue un singolo passo dell'algoritmo;
    con 0 viene invocato una sola volta un kernel che esegue tutto l'algoritmo.
*/
#define KERNEL_SINGLE_STEP 1

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
    CUDA kernel of Bellman-Ford algorithm.
    Each thread executes a relax on a single edge for V-1 times.
*/
__global__ void cuda_bellman_ford (unsigned int n_nodes,
                                   unsigned int n_edges,
                                   Edge* graph,
                                   unsigned int *distances,
                                   unsigned int *mutex) {
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    if(idx < n_edges) {
        #if KERNEL_SINGLE_STEP == 0
        for(unsigned int i=0; i<n_nodes-1; i++) {
            // relax the edge (u,v)
            const unsigned int u = graph[idx].start_node;
            const unsigned int v = graph[idx].end_node;
            // controllo overflow-safe
            if(distances[v] > distances[u] && distances[v]-distances[u] > graph[idx].weight) {
                while(!atomicCAS(&mutex[ v ], 0, 1)) ;

                //mutex[ graph[idx].end_node ] = 1;
                if(distances[v] > distances[u] && distances[v]-distances[u] > graph[idx].weight) {
                    distances[v] = distances[u] + graph[idx].weight;
                }
                mutex[v] = 0;
            }
        }
        #elif KERNEL_SINGLE_STEP == 1
        // relax the edge (u,v)
        const unsigned int u = graph[idx].start_node;
        const unsigned int v = graph[idx].end_node;
        // controllo overflow-safe
        if(distances[v] > distances[u] && distances[v]-distances[u] > graph[idx].weight) {
            while(!atomicCAS(&mutex[ v ], 0, 1)) ;

            //mutex[ graph[idx].end_node ] = 1;
            if(distances[v] > distances[u] && distances[v]-distances[u] > graph[idx].weight) {
                distances[v] = distances[u] + graph[idx].weight;
            }
            mutex[v] = 0;
        }
        #endif
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
    size_t sz_mutex = n_nodes * sizeof(unsigned int);

    unsigned int *h_mutex = (unsigned int*) malloc(sz_mutex);
    assert(h_mutex);
    unsigned int *d_mutex;

    for(unsigned int i=0; i<n_nodes; i++) {
        h_mutex[i] = 0;
    }

    Edge* d_graph;

    unsigned int *d_distances;
    unsigned int *h_distances = (unsigned int*) malloc(sz_distances);
    assert(h_distances);

    for(unsigned int i=0; i<n_nodes; i++) {
        h_distances[i] = 0xffffffff;
    }
    h_distances[source] = 0;

    // malloc and copy of the distances array
    cudaSafeCall( cudaMalloc((void**)&d_distances, sz_distances) );
    cudaSafeCall( cudaMemcpy(d_distances, h_distances, sz_distances, cudaMemcpyHostToDevice) );

    // malloc and copy of the graph
    cudaSafeCall( cudaMalloc((void**)&d_graph, sz_graph) );
    cudaSafeCall( cudaMemcpy(d_graph, h_graph, sz_graph, cudaMemcpyHostToDevice) );

    // preparo le mutex
    cudaSafeCall( cudaMalloc((void**)&d_mutex, sz_mutex) );
    cudaSafeCall( cudaMemcpy(d_mutex, h_mutex, sz_mutex, cudaMemcpyHostToDevice) );

    #if KERNEL_SINGLE_STEP == 0
    // kernel invocation
    cuda_bellman_ford<<< (n_edges+BLKDIM-1) / BLKDIM, BLKDIM >>>(n_nodes, n_edges, d_graph, d_distances, d_mutex);
    cudaCheckError();

    #elif KERNEL_SINGLE_STEP == 1

    for(unsigned int i=0; i<n_nodes-1; i++) {
        cuda_bellman_ford<<< (n_edges+BLKDIM-1) / BLKDIM, BLKDIM >>>(n_nodes, n_edges, d_graph, d_distances, d_mutex);
        cudaCheckError();
    }
    #endif

    // copy-back of the result
    cudaSafeCall( cudaMemcpy(h_distances, d_distances, sz_distances, cudaMemcpyDeviceToHost) );

    // deallocation
    cudaFree(d_graph);
    cudaFree(d_distances);
    cudaFree(d_mutex);

    return h_distances;
}

int main ( void ) {

    Edge *graph;
    unsigned int nodes, edges;
    unsigned int *result;

    fprintf(stderr, "Reading input graph...");
    graph = read_graph(&nodes, &edges);
    fprintf(stderr, "done\n");

    #if KERNEL_SINGLE_STEP == 0
    fprintf(stderr, "Computing Bellman-Ford (no single step)...");
    #elif KERNEL_SINGLE_STEP == 1
    fprintf(stderr, "Computing Bellman-Ford (single step)...");
    #endif
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