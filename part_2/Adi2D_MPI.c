#include <mpi.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#define  Max(a,b) ((a)>(b)?(a):(b))

#define  N 8192
double maxeps = 0.1e-7;
int itmax = 100;

double eps;

void relax();
void init();
void verify(); 

int world_rank;
int world_size;
int world_start, world_end;

void calc_start_end(int *start, int *end, int rank) {
    int for_one = N / world_size;
    if (rank != 0){
        *start = rank * for_one;
    } else {
        *start = 1;
    }
    if (rank != world_size - 1){
        *end = (rank + 1) * for_one;
    } else {
        *end = N - 1;
    }
}

double **get_mem(int size) {
    double **A = malloc(size * sizeof(*A));
    for (int i = 0; i < size; i++) {
        A[i] = malloc(N * sizeof(*A[i]));  
    }

    return A;
}

void free_mem(double **A, int size) {
    for (int i = 0; i < size; i++) {
        free(A[i]);
    }
    free(A);
}

int main(int an, char **as)
{
    MPI_Init(NULL, NULL);

    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    
    calc_start_end(&world_start, &world_end, world_rank);
    int size = world_end - world_start + 2;
    double **A = get_mem(size);

    double time_begin = MPI_Wtime();

    fprintf(stderr, "world size: %d, world rank: %d, start: %d, end: %d\n", world_size, world_rank, world_start, world_end);
    init(A, size);
    MPI_Barrier(MPI_COMM_WORLD);
	for(int it = 1; it <= itmax; it++) {
		eps = 0.;
		relax(A, size);
        if (world_rank == 0) {
            printf( "it=%4i   eps=%f\n", it,eps);
            if (eps < maxeps) break;
        }
    }
	verify(A, size);

    double all_time = MPI_Wtime() - time_begin;

    if (world_rank == 0) {
        printf("size = %d\ntime = %lf\n", world_size, all_time);
    }
    
    free_mem(A, size);

    MPI_Finalize();
	return 0;
}

void init(double **A, int size) {
    for (int i = 0; i < size; i++) {
	    for(int j = 1; j < N - 1; j++) {
            if (world_start == 1 && i == 0) A[i][j] = 0;
            else if (world_end == N - 1 && i == size - 1) A[i][j] = 0;
            else A[i][j] = (1. + i + world_start - 1 + j) ;
        }
    }
} 


void relax(double **A, int size) {
	if (world_rank != 0) {
        MPI_Recv(A[0], N, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    for(int i = 1; i < size - 1; i++) {
	    for(int j = 1; j < N - 1; j++) {
		    A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
        }
	}

    if (world_rank != world_size - 1) {
        MPI_Request req;
        MPI_Isend(A[size - 2], N, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }
    double local_eps = 0;
	
    for(int i = 1; i < size - 1; i++) {
        for(int j = 1; j < N - 1; j++) {
		    double e = A[i][j];
		    A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
		    local_eps = Max(local_eps, fabs(e - A[i][j]));
        }
	}

    if (world_rank != 0) {
        MPI_Request req;
        MPI_Isend(A[1], N, MPI_DOUBLE, world_rank - 1, 0, MPI_COMM_WORLD, &req);
        MPI_Request_free(&req);
    }
    if (world_rank + 1 != world_size) {
        MPI_Recv(A[size - 1], N, MPI_DOUBLE, world_rank + 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
}

void verify(double **A, int size) { 
	double local_sum = 0.;
    double global_sum = 0.;
	for(int i = 1; i < size - 1; i++) {
	    for(int j = 0; j <= N - 1; j++) {
		    local_sum = local_sum + A[i][j] * (i + world_start) * (j + 1) / (N * N);
	    }
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
	
    if (world_rank == 0) {
        printf("  S = %f\n", global_sum);
    }
}
