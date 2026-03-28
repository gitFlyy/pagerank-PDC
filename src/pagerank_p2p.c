/*
 * Scenario 1: Point-to-Point Blocking Communication
 * Uses MPI_Send / MPI_Recv to exchange ghost cell PR values
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "graph_utils.h"

#define DAMPING   0.85
#define THRESHOLD 1e-7
#define MAX_ITER  200

void pagerank_p2p(LocalGraph *lg, int rank, int num_ranks,
                  MPI_Comm comm, int total_N) {

    int N = total_N;

    // Global PR array — each rank keeps a full copy (needed to look up
    // remote PR values after exchange)
    double *global_pr = malloc(N * sizeof(double));

    // Initialize global_pr from local values
    for (int i = 0; i < N; i++) global_pr[i] = 1.0 / N;

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // ── Exchange: rank 0 collects all PR, sends back full array ──────
        // (Manual P2P handshake as required by Scenario 1)
        if (rank == 0) {
            // Copy own values
            for (int li = 0; li < lg->local_n; li++)
                global_pr[lg->global_ids[li]] = lg->pr[li];

            // Receive from all other ranks
            for (int r = 1; r < num_ranks; r++) {
                int count;
                MPI_Recv(&count, 1, MPI_INT, r, 10, comm, MPI_STATUS_IGNORE);
                int    *gids = malloc(count * sizeof(int));
                double *vals = malloc(count * sizeof(double));
                MPI_Recv(gids, count, MPI_INT,    r, 11, comm, MPI_STATUS_IGNORE);
                MPI_Recv(vals, count, MPI_DOUBLE, r, 12, comm, MPI_STATUS_IGNORE);
                for (int i = 0; i < count; i++)
                    global_pr[gids[i]] = vals[i];
                free(gids); free(vals);
            }

            // Broadcast updated global_pr to all ranks
            for (int r = 1; r < num_ranks; r++)
                MPI_Send(global_pr, N, MPI_DOUBLE, r, 13, comm);

        } else {
            // Send my PR values to rank 0
            MPI_Send(&lg->local_n,  1,           MPI_INT,    0, 10, comm);
            MPI_Send(lg->global_ids, lg->local_n, MPI_INT,   0, 11, comm);
            MPI_Send(lg->pr,         lg->local_n, MPI_DOUBLE, 0, 12, comm);
            // Receive full global PR
            MPI_Recv(global_pr, N, MPI_DOUBLE, 0, 13, comm, MPI_STATUS_IGNORE);
        }

        // ── Local PageRank update ─────────────────────────────────────────
        double local_residual = 0.0;

        for (int li = 0; li < lg->local_n; li++) {
            double sum = 0.0;
            int start = lg->in_row_ptr[li];
            int end   = lg->in_row_ptr[li + 1];

            for (int e = start; e < end; e++) {
                int    j    = lg->in_col_idx[e];   // global ID of neighbor
                int    outdeg = lg->out_degree[j];
                if (outdeg > 0)
                    sum += global_pr[j] / outdeg;
            }

            lg->pr_new[li] = (1.0 - DAMPING) / N + DAMPING * sum;
            local_residual += fabs(lg->pr_new[li] - lg->pr[li]);
        }

        // ── Convergence check ─────────────────────────────────────────────
        double global_residual = 0.0;
        MPI_Allreduce(&local_residual, &global_residual,
                      1, MPI_DOUBLE, MPI_SUM, comm);

        // Swap PR arrays
        double *tmp = lg->pr;
        lg->pr      = lg->pr_new;
        lg->pr_new  = tmp;

        print_stats(rank, iter + 1, global_residual);

        if (global_residual < THRESHOLD) {
            if (rank == 0)
                printf("[P2P] Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    free(global_pr);
}
