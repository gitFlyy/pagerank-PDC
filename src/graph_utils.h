#ifndef GRAPH_UTILS_H
#define GRAPH_UTILS_H

#include <mpi.h>
#include <parmetis.h>

typedef struct {
    int  num_vertices;
    int  num_edges;

    int *row_ptr;
    int *col_idx;
    int *in_row_ptr;
    int *in_col_idx;

    int *out_degree;
} Graph;

typedef struct {
    int  local_n;
    int *global_ids;

    int *in_row_ptr;
    int *in_col_idx;

    int *out_degree;

    int  num_internal;
    int  num_boundary;
    int *internal_verts;
    int *boundary_verts;

    int   num_ghosts;
    int  *ghost_global_ids;
    int  *ghost_owner_rank;
    double *ghost_pr;

    double *pr;
    double *pr_new;
} LocalGraph;

Graph* load_graph(const char *edges_file);

void free_graph(Graph *g);

idx_t* partition_graph(Graph *g, int num_ranks, MPI_Comm comm);

LocalGraph* distribute_graph(Graph *g, idx_t *partition,
                              int rank, int num_ranks, MPI_Comm comm);

void classify_vertices(LocalGraph *lg, int rank, int num_ranks);

void free_local_graph(LocalGraph *lg);

void print_stats(int rank, int iter, double residual);

#endif
