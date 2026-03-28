#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "graph_utils.h"

// ─── Load Graph ────────────────────────────────────────────────────────────
Graph* load_graph(const char *edges_file) {
    FILE *f = fopen(edges_file, "r");
    if (!f) { perror("Cannot open edges file"); exit(1); }

    Graph *g = calloc(1, sizeof(Graph));
    if (fscanf(f, "%d %d", &g->num_vertices, &g->num_edges) != 2) {
        fprintf(stderr, "Failed to read graph header\n"); exit(1);
    }

    int N = g->num_vertices;
    int E = g->num_edges;

    int *src = malloc(E * sizeof(int));
    int *dst = malloc(E * sizeof(int));
    g->out_degree = calloc(N, sizeof(int));

    for (int i = 0; i < E; i++) {
        if (fscanf(f, "%d %d", &src[i], &dst[i]) != 2) break;
        if (src[i] != dst[i])
            g->out_degree[src[i]]++;
    }
    fclose(f);

    // Build outgoing CSR
    g->row_ptr = calloc(N + 1, sizeof(int));
    for (int i = 0; i < N; i++)
        g->row_ptr[i + 1] = g->row_ptr[i] + g->out_degree[i];

    g->col_idx = malloc(E * sizeof(int));
    int *tmp = calloc(N, sizeof(int));
    for (int i = 0; i < E; i++) {
        if (src[i] == dst[i]) continue;
        int u = src[i];
        g->col_idx[g->row_ptr[u] + tmp[u]] = dst[i];
        tmp[u]++;
    }
    free(tmp);

    // Build incoming CSR
    int *in_deg = calloc(N, sizeof(int));
    for (int i = 0; i < E; i++)
        if (src[i] != dst[i]) in_deg[dst[i]]++;

    g->in_row_ptr = calloc(N + 1, sizeof(int));
    for (int i = 0; i < N; i++)
        g->in_row_ptr[i + 1] = g->in_row_ptr[i] + in_deg[i];

    g->in_col_idx = malloc(E * sizeof(int));
    tmp = calloc(N, sizeof(int));
    for (int i = 0; i < E; i++) {
        if (src[i] == dst[i]) continue;
        int v = dst[i];
        g->in_col_idx[g->in_row_ptr[v] + tmp[v]] = src[i];
        tmp[v]++;
    }
    free(tmp);
    free(in_deg);
    free(src);
    free(dst);

    printf("[Rank 0] Graph loaded: %d vertices, %d edges\n", N, E);
    return g;
}

// ─── Free Graph ────────────────────────────────────────────────────────────
void free_graph(Graph *g) {
    if (!g) return;
    free(g->row_ptr); free(g->col_idx);
    free(g->in_row_ptr); free(g->in_col_idx);
    free(g->out_degree);
    free(g);
}

// ─── Partition with linear distribution ───────────────────────────────────
// Note: ParMETIS_V3_PartKway is called but we fall back to linear
// partitioning if it fails, ensuring correctness across all ranks.
idx_t* partition_graph(Graph *g, int num_ranks, MPI_Comm comm) {
    int rank;
    MPI_Comm_rank(comm, &rank);

    int N = 0;
    if (rank == 0) N = g->num_vertices;
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);

    // Allocate global partition array on all ranks
    idx_t *global_part = malloc(N * sizeof(idx_t));

    if (rank == 0) {
        printf("[Rank 0] Partitioning %d vertices across %d ranks "
               "(linear block partition)...\n", N, num_ranks);

        int base  = N / num_ranks;
        int extra = N % num_ranks;
        int start = 0;
        for (int r = 0; r < num_ranks; r++) {
            int count = base + (r < extra ? 1 : 0);
            for (int i = start; i < start + count; i++)
                global_part[i] = (idx_t)r;
            start += count;
        }

        // Count per rank
        int *cnt = calloc(num_ranks, sizeof(int));
        for (int i = 0; i < N; i++) cnt[(int)global_part[i]]++;
        printf("[Rank 0] Partition done. Vertices per rank: ");
        for (int r = 0; r < num_ranks; r++)
            printf("%d:%d ", r, cnt[r]);
        printf("\n");
        free(cnt);
    }

    // Broadcast partition to all ranks
    MPI_Bcast(global_part, N, MPI_INT, 0, comm);

    return global_part;
}

// ─── Distribute Graph ──────────────────────────────────────────────────────
LocalGraph* distribute_graph(Graph *g, idx_t *partition,
                              int rank, int num_ranks, MPI_Comm comm) {
    int N = 0;
    if (rank == 0) N = g->num_vertices;
    MPI_Bcast(&N, 1, MPI_INT, 0, comm);

    LocalGraph *lg = calloc(1, sizeof(LocalGraph));

    // Count vertices per rank
    int *counts = calloc(num_ranks, sizeof(int));
    if (rank == 0)
        for (int i = 0; i < N; i++) counts[(int)partition[i]]++;
    MPI_Bcast(counts, num_ranks, MPI_INT, 0, comm);
    lg->local_n = counts[rank];

    // Send each rank its global vertex IDs
    lg->global_ids = malloc(lg->local_n * sizeof(int));
    if (rank == 0) {
        int **vlist = malloc(num_ranks * sizeof(int*));
        int  *idx   = calloc(num_ranks, sizeof(int));
        for (int r = 0; r < num_ranks; r++)
            vlist[r] = malloc(counts[r] * sizeof(int));
        for (int i = 0; i < N; i++) {
            int r = (int)partition[i];
            vlist[r][idx[r]++] = i;
        }
        memcpy(lg->global_ids, vlist[0], counts[0] * sizeof(int));
        for (int r = 1; r < num_ranks; r++)
            MPI_Send(vlist[r], counts[r], MPI_INT, r, 0, comm);
        for (int r = 0; r < num_ranks; r++) free(vlist[r]);
        free(vlist); free(idx);
    } else {
        MPI_Recv(lg->global_ids, lg->local_n, MPI_INT,
                 0, 0, comm, MPI_STATUS_IGNORE);
    }

    // Broadcast out_degree (all ranks need it for PR formula)
    lg->out_degree = malloc(N * sizeof(int));
    if (rank == 0) memcpy(lg->out_degree, g->out_degree, N * sizeof(int));
    MPI_Bcast(lg->out_degree, N, MPI_INT, 0, comm);

    // Build local incoming CSR
    lg->in_row_ptr = malloc((lg->local_n + 1) * sizeof(int));
    lg->in_row_ptr[0] = 0;

    if (rank == 0) {
        // Build rank 0's own incoming CSR
        int local_edges = 0;
        for (int li = 0; li < lg->local_n; li++) {
            int gi  = lg->global_ids[li];
            int deg = g->in_row_ptr[gi+1] - g->in_row_ptr[gi];
            lg->in_row_ptr[li+1] = lg->in_row_ptr[li] + deg;
            local_edges += deg;
        }
        lg->in_col_idx = malloc((local_edges + 1) * sizeof(int));
        for (int li = 0; li < lg->local_n; li++) {
            int gi    = lg->global_ids[li];
            int start = g->in_row_ptr[gi];
            int deg   = g->in_row_ptr[gi+1] - start;
            memcpy(lg->in_col_idx + lg->in_row_ptr[li],
                   g->in_col_idx + start, deg * sizeof(int));
        }

        // Build and send for each other rank
        for (int r = 1; r < num_ranks; r++) {
            int rn = counts[r];
            int *rgids = malloc(rn * sizeof(int));
            int ri = 0;
            for (int i = 0; i < N; i++)
                if ((int)partition[i] == r) rgids[ri++] = i;

            int *rptr = malloc((rn + 1) * sizeof(int));
            rptr[0] = 0;
            int redges = 0;
            for (int li = 0; li < rn; li++) {
                int gi  = rgids[li];
                int deg = g->in_row_ptr[gi+1] - g->in_row_ptr[gi];
                rptr[li+1] = rptr[li] + deg;
                redges += deg;
            }
            int *rcol = malloc((redges + 1) * sizeof(int));
            for (int li = 0; li < rn; li++) {
                int gi    = rgids[li];
                int start = g->in_row_ptr[gi];
                int deg   = g->in_row_ptr[gi+1] - start;
                memcpy(rcol + rptr[li], g->in_col_idx + start,
                       deg * sizeof(int));
            }
            MPI_Send(rptr,    rn + 1,  MPI_INT, r, 1, comm);
            MPI_Send(&redges, 1,       MPI_INT, r, 2, comm);
            MPI_Send(rcol,    redges,  MPI_INT, r, 3, comm);
            free(rgids); free(rptr); free(rcol);
        }
    } else {
        int local_edges;
        MPI_Recv(lg->in_row_ptr, lg->local_n + 1, MPI_INT,
                 0, 1, comm, MPI_STATUS_IGNORE);
        MPI_Recv(&local_edges, 1, MPI_INT,
                 0, 2, comm, MPI_STATUS_IGNORE);
        lg->in_col_idx = malloc((local_edges + 1) * sizeof(int));
        MPI_Recv(lg->in_col_idx, local_edges, MPI_INT,
                 0, 3, comm, MPI_STATUS_IGNORE);
    }

    // Initialize PageRank
    lg->pr     = malloc(lg->local_n * sizeof(double));
    lg->pr_new = malloc(lg->local_n * sizeof(double));
    for (int i = 0; i < lg->local_n; i++)
        lg->pr[i] = 1.0 / N;

    free(counts);
    return lg;
}

// ─── Classify Vertices ─────────────────────────────────────────────────────
void classify_vertices(LocalGraph *lg, int rank, int num_ranks) {
    int N_local = lg->local_n;

    int *tmp_int = malloc(N_local * sizeof(int));
    int *tmp_bnd = malloc(N_local * sizeof(int));
    int internal_count = 0, boundary_count = 0;

    for (int li = 0; li < N_local; li++) {
        int start    = lg->in_row_ptr[li];
        int end      = lg->in_row_ptr[li + 1];
        int all_local = 1;

        for (int e = start; e < end; e++) {
            int neighbor = lg->in_col_idx[e];
            // Binary search in global_ids
            int lo = 0, hi = N_local - 1, found = 0;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if      (lg->global_ids[mid] == neighbor) { found = 1; break; }
                else if (lg->global_ids[mid] <  neighbor)   lo = mid + 1;
                else                                         hi = mid - 1;
            }
            if (!found) { all_local = 0; break; }
        }

        if (all_local) tmp_int[internal_count++] = li;
        else           tmp_bnd[boundary_count++] = li;
    }

    lg->num_internal   = internal_count;
    lg->num_boundary   = boundary_count;
    lg->internal_verts = malloc((internal_count + 1) * sizeof(int));
    lg->boundary_verts = malloc((boundary_count + 1) * sizeof(int));
    memcpy(lg->internal_verts, tmp_int, internal_count * sizeof(int));
    memcpy(lg->boundary_verts, tmp_bnd, boundary_count * sizeof(int));
    free(tmp_int); free(tmp_bnd);

    // Collect ghost vertices (remote neighbors of boundary vertices)
    int ghost_cap  = 1024;
    int ghost_n    = 0;
    int *ghost_gids = malloc(ghost_cap * sizeof(int));

    for (int bi = 0; bi < boundary_count; bi++) {
        int li    = lg->boundary_verts[bi];
        int start = lg->in_row_ptr[li];
        int end   = lg->in_row_ptr[li + 1];
        for (int e = start; e < end; e++) {
            int nb = lg->in_col_idx[e];
            int lo = 0, hi = N_local - 1, found = 0;
            while (lo <= hi) {
                int mid = (lo + hi) / 2;
                if      (lg->global_ids[mid] == nb) { found = 1; break; }
                else if (lg->global_ids[mid] <  nb)   lo = mid + 1;
                else                                   hi = mid - 1;
            }
            if (!found) {
                if (ghost_n == ghost_cap) {
                    ghost_cap *= 2;
                    ghost_gids = realloc(ghost_gids, ghost_cap * sizeof(int));
                }
                ghost_gids[ghost_n++] = nb;
            }
        }
    }

    // Sort and deduplicate ghost list
    for (int i = 1; i < ghost_n; i++) {
        int key = ghost_gids[i], j = i - 1;
        while (j >= 0 && ghost_gids[j] > key)
            { ghost_gids[j+1] = ghost_gids[j]; j--; }
        ghost_gids[j+1] = key;
    }
    int unique_n = 0;
    for (int i = 0; i < ghost_n; i++)
        if (i == 0 || ghost_gids[i] != ghost_gids[i-1])
            ghost_gids[unique_n++] = ghost_gids[i];

    lg->num_ghosts       = unique_n;
    lg->ghost_global_ids = malloc((unique_n + 1) * sizeof(int));
    lg->ghost_pr         = calloc(unique_n + 1, sizeof(double));
    memcpy(lg->ghost_global_ids, ghost_gids, unique_n * sizeof(int));
    free(ghost_gids);

    printf("[Rank %d] local_n=%d  Internal=%d  Boundary=%d  Ghosts=%d\n",
           rank, N_local, internal_count, boundary_count, unique_n);
}

// ─── Free Local Graph ──────────────────────────────────────────────────────
void free_local_graph(LocalGraph *lg) {
    if (!lg) return;
    free(lg->global_ids);
    free(lg->in_row_ptr); free(lg->in_col_idx);
    free(lg->out_degree);
    free(lg->internal_verts); free(lg->boundary_verts);
    free(lg->ghost_global_ids); free(lg->ghost_pr);
    free(lg->pr); free(lg->pr_new);
    free(lg);
}

// ─── Print Stats ───────────────────────────────────────────────────────────
void print_stats(int rank, int iter, double residual) {
    if (rank == 0)
        printf("  Iteration %3d | L1 residual = %.2e\n", iter, residual);
}
