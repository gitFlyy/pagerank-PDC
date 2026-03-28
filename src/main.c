
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>
#include "graph_utils.h"

void pagerank_p2p        (LocalGraph *lg, int rank, int num_ranks,
                          MPI_Comm comm, int total_N);
void pagerank_collective (LocalGraph *lg, int rank, int num_ranks,
                          MPI_Comm comm, int total_N);
void pagerank_async      (LocalGraph *lg, int rank, int num_ranks,
                          MPI_Comm comm, int total_N);

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, num_ranks;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &num_ranks);

    if (argc < 2) {
        if (rank == 0)
            fprintf(stderr, "Usage: %s <edges_file>\n", argv[0]);
        MPI_Finalize();
        return 1;
    }

    Graph *g = NULL;
    if (rank == 0) {
        printf("=== PageRank Engine (Scenario %d) ===\n", SCENARIO);
        printf("[Rank 0] Loading graph from %s\n", argv[1]);
        g = load_graph(argv[1]);
    }

    idx_t *partition = NULL;
    int total_N = 0;

    if (rank == 0) total_N = g->num_vertices;
    MPI_Bcast(&total_N, 1, MPI_INT, 0, MPI_COMM_WORLD);

    partition = partition_graph(g, num_ranks, MPI_COMM_WORLD);

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) printf("[Rank 0] Distributing graph...\n");

    LocalGraph *lg = distribute_graph(g, partition,
                                      rank, num_ranks, MPI_COMM_WORLD);

    if (rank == 0) free_graph(g);
    free(partition);

    MPI_Barrier(MPI_COMM_WORLD);
    classify_vertices(lg, rank, num_ranks);

    MPI_Barrier(MPI_COMM_WORLD);
    double t_start = MPI_Wtime();

#if SCENARIO == 1
    if (rank == 0) printf("\n[Scenario 1] P2P Blocking (MPI_Send/Recv)\n");
    pagerank_p2p(lg, rank, num_ranks, MPI_COMM_WORLD, total_N);

#elif SCENARIO == 2
    if (rank == 0) printf("\n[Scenario 2] Collectives (MPI_Allgatherv)\n");
    pagerank_collective(lg, rank, num_ranks, MPI_COMM_WORLD, total_N);

#elif SCENARIO == 3
    if (rank == 0) printf("\n[Scenario 3] Async Overlap (MPI_Isend/Irecv)\n");
    pagerank_async(lg, rank, num_ranks, MPI_COMM_WORLD, total_N);
#endif

    double t_end = MPI_Wtime();

    MPI_Barrier(MPI_COMM_WORLD);
    if (rank == 0) {
        printf("\n=== Total wall time: %.4f seconds ===\n", t_end - t_start);
    }

    int *all_counts   = NULL;
    int *displs       = NULL;
    double *global_pr = NULL;
    int *global_gids  = NULL;

    int *counts_send = &lg->local_n;
    if (rank == 0) {
        all_counts  = malloc(num_ranks * sizeof(int));
        displs      = malloc(num_ranks * sizeof(int));
    }
    MPI_Gather(&lg->local_n, 1, MPI_INT,
               all_counts, 1, MPI_INT, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        displs[0] = 0;
        for (int i = 1; i < num_ranks; i++)
            displs[i] = displs[i-1] + all_counts[i-1];
        global_pr   = malloc(total_N * sizeof(double));
        global_gids = malloc(total_N * sizeof(int));
    }

    MPI_Gatherv(lg->pr, lg->local_n, MPI_DOUBLE,
                global_pr, all_counts, displs, MPI_DOUBLE,
                0, MPI_COMM_WORLD);
    MPI_Gatherv(lg->global_ids, lg->local_n, MPI_INT,
                global_gids, all_counts, displs, MPI_INT,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("\n--- Top 10 Vertices by PageRank ---\n");
        for (int t = 0; t < 10; t++) {
            double best = -1.0;
            int    best_i = 0;
            for (int i = 0; i < total_N; i++)
                if (global_pr[i] > best)
                    { best = global_pr[i]; best_i = i; }
            printf("  #%2d  vertex %7d  PR = %.8f\n",
                   t+1, global_gids[best_i], best);
            global_pr[best_i] = -1.0;
        }
        free(all_counts); free(displs);
        free(global_pr);  free(global_gids);
    }

    free_local_graph(lg);
    MPI_Finalize();
    return 0;
}
