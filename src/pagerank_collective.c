
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "graph_utils.h"

#define DAMPING   0.85
#define THRESHOLD 1e-7
#define MAX_ITER  200

void pagerank_collective(LocalGraph *lg, int rank, int num_ranks,
                         MPI_Comm comm, int total_N) {

    int N = total_N;
    double *global_pr = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) global_pr[i] = 1.0 / N;
    int *counts = malloc(num_ranks * sizeof(int));
    int *displs = malloc(num_ranks * sizeof(int));

    // Gather how many vertices each rank owns
    MPI_Allgather(&lg->local_n, 1, MPI_INT,
                  counts, 1, MPI_INT, comm);

    displs[0] = 0;
    for (int i = 1; i < num_ranks; i++)
        displs[i] = displs[i-1] + counts[i-1];
    int *all_gids = malloc(N * sizeof(int));
    MPI_Allgatherv(lg->global_ids, lg->local_n, MPI_INT,
                   all_gids, counts, displs, MPI_INT, comm);


    for (int iter = 0; iter < MAX_ITER; iter++) {

        double *gathered_pr = malloc(N * sizeof(double));

        MPI_Allgatherv(lg->pr, lg->local_n, MPI_DOUBLE,
                       gathered_pr, counts, displs, MPI_DOUBLE, comm);

        // Reconstruct global_pr indexed by global vertex ID
        for (int i = 0; i < N; i++)
            global_pr[all_gids[i]] = gathered_pr[i];

        free(gathered_pr);

        // ── Local PageRank update ────
        double local_residual = 0.0;

        for (int li = 0; li < lg->local_n; li++) {
            double sum = 0.0;
            int start = lg->in_row_ptr[li];
            int end   = lg->in_row_ptr[li + 1];

            for (int e = start; e < end; e++) {
                int j     = lg->in_col_idx[e];
                int outdeg = lg->out_degree[j];
                if (outdeg > 0)
                    sum += global_pr[j] / outdeg;
            }

            lg->pr_new[li] = (1.0 - DAMPING) / N + DAMPING * sum;
            local_residual += fabs(lg->pr_new[li] - lg->pr[li]);
        }

        // ── Global convergence check ───
        double global_residual = 0.0;
        MPI_Allreduce(&local_residual, &global_residual,
                      1, MPI_DOUBLE, MPI_SUM, comm);

        // Swap arrays
        double *tmp = lg->pr;
        lg->pr      = lg->pr_new;
        lg->pr_new  = tmp;

        print_stats(rank, iter + 1, global_residual);

        if (global_residual < THRESHOLD) {
            if (rank == 0)
                printf("[Collective] Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    free(global_pr);
    free(counts);
    free(displs);
    free(all_gids);
}
