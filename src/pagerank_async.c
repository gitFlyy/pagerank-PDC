/*
 * Scenario 3: Asynchronous Communication-Computation Overlap
 * Uses MPI_Isend/Irecv to hide communication latency behind
 * computation of Internal Vertices (those needing no remote data)
 */

#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <mpi.h>
#include "graph_utils.h"

#define DAMPING   0.85
#define THRESHOLD 1e-7
#define MAX_ITER  200

void pagerank_async(LocalGraph *lg, int rank, int num_ranks,
                    MPI_Comm comm, int total_N) {

    int N = total_N;

    double *global_pr = malloc(N * sizeof(double));
    for (int i = 0; i < N; i++) global_pr[i] = 1.0 / N;

    // Setup Allgatherv metadata (same as Scenario 2)
    int *counts  = malloc(num_ranks * sizeof(int));
    int *displs  = malloc(num_ranks * sizeof(int));
    MPI_Allgather(&lg->local_n, 1, MPI_INT, counts, 1, MPI_INT, comm);
    displs[0] = 0;
    for (int i = 1; i < num_ranks; i++)
        displs[i] = displs[i-1] + counts[i-1];

    int *all_gids = malloc(N * sizeof(int));
    MPI_Allgatherv(lg->global_ids, lg->local_n, MPI_INT,
                   all_gids, counts, displs, MPI_INT, comm);

    // Buffers for non-blocking sends/receives
    double **send_bufs = malloc(num_ranks * sizeof(double*));
    double **recv_bufs = malloc(num_ranks * sizeof(double*));
    for (int r = 0; r < num_ranks; r++) {
        send_bufs[r] = malloc(counts[r] * sizeof(double));
        recv_bufs[r] = malloc(counts[r] * sizeof(double));
    }

    MPI_Request *requests = malloc(2 * num_ranks * sizeof(MPI_Request));

    for (int iter = 0; iter < MAX_ITER; iter++) {

        // ── Phase 1: Post all non-blocking sends and receives ─────────────
        int req_count = 0;

        // Post Irecv from all ranks
        for (int r = 0; r < num_ranks; r++) {
            MPI_Irecv(recv_bufs[r], counts[r], MPI_DOUBLE,
                      r, 20, comm, &requests[req_count++]);
        }

        // Fill send buffer with my current PR values and post Isend
        for (int li = 0; li < lg->local_n; li++)
            send_bufs[rank][li] = lg->pr[li];

        for (int r = 0; r < num_ranks; r++) {
            MPI_Isend(send_bufs[rank], lg->local_n, MPI_DOUBLE,
                      r, 20, comm, &requests[req_count++]);
        }

        // ── Phase 2: Compute INTERNAL vertices while comms in flight ──────
        // This is the key overlap: internal vertices don't need remote data
        double local_residual = 0.0;

        for (int ii = 0; ii < lg->num_internal; ii++) {
            int li    = lg->internal_verts[ii];
            double sum = 0.0;
            int start  = lg->in_row_ptr[li];
            int end    = lg->in_row_ptr[li + 1];

            for (int e = start; e < end; e++) {
                int j      = lg->in_col_idx[e];
                int outdeg = lg->out_degree[j];
                if (outdeg > 0)
                    sum += global_pr[j] / outdeg;
            }
            lg->pr_new[li] = (1.0 - DAMPING) / N + DAMPING * sum;
            local_residual += fabs(lg->pr_new[li] - lg->pr[li]);
        }

        // ── Phase 3: Wait for all communications to complete ──────────────
        MPI_Waitall(req_count, requests, MPI_STATUSES_IGNORE);

        // Reconstruct global_pr from received buffers
        int offset = 0;
        for (int r = 0; r < num_ranks; r++) {
            for (int i = 0; i < counts[r]; i++)
                global_pr[all_gids[offset + i]] = recv_bufs[r][i];
            offset += counts[r];
        }

        // ── Phase 4: Compute BOUNDARY vertices (comms now done) ───────────
        for (int bi = 0; bi < lg->num_boundary; bi++) {
            int li     = lg->boundary_verts[bi];
            double sum = 0.0;
            int start  = lg->in_row_ptr[li];
            int end    = lg->in_row_ptr[li + 1];

            for (int e = start; e < end; e++) {
                int j      = lg->in_col_idx[e];
                int outdeg = lg->out_degree[j];
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

        double *tmp = lg->pr;
        lg->pr      = lg->pr_new;
        lg->pr_new  = tmp;

        print_stats(rank, iter + 1, global_residual);

        if (global_residual < THRESHOLD) {
            if (rank == 0)
                printf("[Async] Converged at iteration %d\n", iter + 1);
            break;
        }
    }

    // Cleanup
    for (int r = 0; r < num_ranks; r++) {
        free(send_bufs[r]);
        free(recv_bufs[r]);
    }
    free(send_bufs); free(recv_bufs);
    free(requests);
    free(global_pr);
    free(counts); free(displs);
    free(all_gids);
}
