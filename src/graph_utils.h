#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <mpi.h>
#include <parmetis.h>

// ─── Graph Structure ───────────────────────────────────────────────────────
// Stores the full directed graph in CSR (Compressed Sparse Row) format
typedef struct {
    int  num_vertices;      // Total vertices in full graph (N)
    int  num_edges;         // Total directed edges

    // CSR arrays for directed edges (used for PageRank)
    int *row_ptr;           // row_ptr[i]..row_ptr[i+1] = outgoing edges of i
    int *col_idx;           // destination vertices
    int *in_row_ptr;        // same but for incoming edges
    int *in_col_idx;

    int *out_degree;        // out_degree[i] = number of outgoing links from i
} Graph;

// ─── Local Partition Structure ─────────────────────────────────────────────
// What each MPI rank owns after ParMETIS partitioning
typedef struct {
    int  local_n;           // number of vertices owned by this rank
    int *global_ids;        // global_ids[i] = global vertex ID of local vertex i

    // Local CSR for incoming edges (for PageRank update)
    int *in_row_ptr;
    int *in_col_idx;

    int *out_degree;        // out_degree per vertex (global array, replicated)

    // Vertex classification
    int  num_internal;      // vertices with all incoming edges local
    int  num_boundary;      // vertices needing remote data
    int *internal_verts;    // list of internal vertex local IDs
    int *boundary_verts;    // list of boundary vertex local IDs

    // Ghost cell info
    int   num_ghosts;       // how many remote vertices we need
    int  *ghost_global_ids; // their global IDs
    int  *ghost_owner_rank; // which rank owns each ghost
    double *ghost_pr;       // buffered PageRank values from remote ranks

    // PageRank arrays
    double *pr;             // current PR values (local_n)
    double *pr_new;         // next iteration PR values (local_n)
} LocalGraph;

// ─── Function Declarations ─────────────────────────────────────────────────

// Load full graph from .edges file (only rank 0 does this)
Graph* load_graph(const char *edges_file);

// Free graph memory
void free_graph(Graph *g);

// Partition graph using ParMETIS, returns partition array (size N)
// partition[i] = which MPI rank owns vertex i
idx_t* partition_graph(Graph *g, int num_ranks, MPI_Comm comm);

// Distribute graph to all ranks based on partition
LocalGraph* distribute_graph(Graph *g, idx_t *partition,
                              int rank, int num_ranks, MPI_Comm comm);

// Classify vertices into internal/boundary and set up ghost cells
void classify_vertices(LocalGraph *lg, int rank, int num_ranks);

// Free local graph memory
void free_local_graph(LocalGraph *lg);

// Print convergence info
void print_stats(int rank, int iter, double residual);

#endif // GRAPH_UTILS_H

