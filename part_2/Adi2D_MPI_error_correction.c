#include <mpi.h>
#include <mpi-ext.h>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>
#include <signal.h>
#include <unistd.h>
#define  Max(a,b) ((a)>(b)?(a):(b))
#define P() printf("line: %d\n", __LINE__);

#define  N 8192
#define SAVE_RANGE 5
int last_save_iter = 1;
double maxeps = 0.1e-7;
int itmax = 100;
int additional_procs = 2;

double eps;
double **A;

void relax();
double update();
void init();
void verify(); 
void safe_data();
void restore_data();


int world_rank;
int world_size;
int size;
int world_start, world_end;

bool error = false;
bool was_error = false;
MPI_Comm main_comm = MPI_COMM_WORLD;

void print(double **mat) {
    for (int i = 0; i < size; i++) {
        for (int j = 0; j < N; j++) {
            printf("%lf ", mat[i][j]);
        }
        printf("\n");
    }
    fflush(stdout);
}

void calc_start_end(int *start, int *end, int rank) {
    int for_one = N / world_size;
    if (rank >= world_size) {
        start = 0;
        end = 0;
        return;
    }
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

static void verbose_errhandler(MPI_Comm* pcomm, int* perr, ...) {
    MPI_Comm comm= *pcomm;
    int err = *perr;
    char errstr[MPI_MAX_ERROR_STRING];
    int i, rank, comm_size, nf, len, eclass;
    MPI_Group group_c, group_f;
    int*ranks_gc, *ranks_gf;
    
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &comm_size);
    MPIX_Comm_failure_ack(comm);
    MPIX_Comm_failure_get_acked(comm, &group_f);
    
    MPI_Group_size(group_f, &nf);
    MPI_Error_string(err, errstr, &len);
    
    printf("Rank %d / %d: Notified of error %s. %d found dead: { ",rank, comm_size, errstr, nf);
    
    ranks_gf = (int*)malloc(nf* sizeof(int));
    ranks_gc = (int*)malloc(nf* sizeof(int));
    
    MPI_Comm_group(comm, &group_c);
    
    for(i= 0; i < nf; i++)
        ranks_gf[i] = i;
    
    MPI_Group_translate_ranks(group_f, nf, ranks_gf,group_c, ranks_gc);
    
    for(i= 0; i < nf; i++)
        printf("%d ", ranks_gc[i]);
    printf("}\n");
    
    MPIX_Comm_shrink(comm, &main_comm);
    int temp = world_rank; 
    
    MPI_Comm_rank(main_comm, &world_rank);
    MPI_Comm_size(main_comm, &world_size);
    printf("rank: %d -> %d\n", temp, world_rank);
    fflush(stdout);

    additional_procs -= nf;

    error = true;
    world_size -= additional_procs;
    
    if (world_start != world_end) {
        free_mem(A, size);
    }
    
    calc_start_end(&world_start, &world_end, world_rank);
    size = world_end - world_start + 2;
    
    if (world_start != world_end) {
        A = get_mem(size);
    }
    

    fflush(stdout);
    free(ranks_gf); 
    free(ranks_gc);
}


void save_matrix() {
    char name[100];
    sprintf(name, "matrix_%d", world_rank);
    FILE *file = fopen(name, "wb");
    for (int i = 0;i < size; i++) {
        fwrite(&A[i][0], sizeof(double), N, file);
    }
    fclose(file);
    fprintf(stdout, "save matrix %s, world rank: %d size: %d\n",name, world_rank, size);
}


void load_matrix() {
    char name[100];
    sprintf(name, "matrix_%d", world_rank);
    FILE *file = fopen(name, "rb");
    for (int i = 0;i < size; i++) {
        fread(&A[i][0], sizeof(double), N, file);
    }
    fclose(file);
    fprintf(stdout, "load matrix: %s, world rank: %d size: %d\n",name, world_rank, size);
}


int main(int an, char **as) {
    MPI_Init(&an, &as);
    
    MPI_Comm parent_comm, intercom_comm;
    MPI_Comm_get_parent(&parent_comm);

    if (parent_comm == MPI_COMM_NULL) {
        MPI_Comm_spawn("./a.out", 
                MPI_ARGV_NULL, 
                additional_procs, 
                MPI_INFO_NULL, 0, 
                main_comm, 
                &intercom_comm, 
                MPI_ERRCODES_IGNORE);
    } else {
        intercom_comm = parent_comm;
    }

    MPI_Intercomm_merge(intercom_comm, (parent_comm == MPI_COMM_NULL ? 0 : 1), &main_comm);
    
    MPI_Comm_rank(main_comm, &world_rank);
    MPI_Comm_size(main_comm, &world_size);
    
    MPI_Errhandler errh;
    MPI_Comm_create_errhandler(verbose_errhandler, &errh);
    MPI_Comm_set_errhandler(main_comm, errh);
    
    world_size -= additional_procs;

    calc_start_end(&world_start, &world_end, world_rank);
    size = world_end - world_start + 2;
   
    if (world_start != world_end) {
        A = get_mem(size);
        if (A == NULL) {
            fprintf(stdout, "WTF: %d\n", world_rank);
        }
        fprintf(stdout, "world size: %d, world rank: %d, start: %d, end: %d\n", world_size, world_rank, world_start, world_end);
    }
    MPI_Barrier(main_comm);
    
    if (error && world_rank == 0) {
        printf("add procs: %d\n", additional_procs);
    }

    double time_begin = MPI_Wtime();

    if (world_start != world_end) {
        init(A, size);
    }
    MPI_Barrier(main_comm);
	for(int it = 1; it <= itmax; it++) {
        MPI_Barrier(main_comm);
        
        if (error) {
            if (world_start != world_end)
                load_matrix();
            it = last_save_iter;
            error = false;
            was_error = true;
        }
        
        if (it == SAVE_RANGE + last_save_iter) {
            if (world_start != world_end) {
                save_matrix();
            }
            last_save_iter = it;
        }
        eps = 0.;
		relax(A, size);
        
        if (error) {
            continue;
        }
        if (world_rank == 0 && (it == 10 || it == 90) && !was_error) {
            raise(SIGKILL);
        }
        double local_eps = update();
        
        if (error) {
            continue;
        }
        //fprintf(stdout, "before barrier: %d iter: %d line: %d\n", world_rank, it, __LINE__); 

        MPI_Barrier(main_comm);
        if (error) {
            continue;
        }
        MPI_Reduce(&local_eps, &eps, 1, MPI_DOUBLE, MPI_MAX, 0, main_comm);

        if (world_rank == 0) {
            printf( "it=%4i   eps=%f  last_save=%d\n", it, eps, last_save_iter);
            fflush(stdout);
            if (eps < maxeps) break;
        }
    }
    if (error) {
        error = false;
    }
	verify(A, size);
    if (error) {
        verify(A, size);
    }

    double all_time = MPI_Wtime() - time_begin;

    if (world_rank == 0) {
        printf("size = %d\ntime = %lf\n", world_size, all_time);
    }
    if (world_start != world_end) {
        free_mem(A, size);
    }
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
	if (world_start == world_end) {
        return;
    }
    
    if (world_rank != 0) {
        MPI_Recv(A[0], N, MPI_DOUBLE, world_rank - 1, 0, main_comm, MPI_STATUS_IGNORE);
    }
    for(int i = 1; i < size - 1; i++) {
	    for(int j = 1; j < N - 1; j++) {
		    A[i][j] = (A[i - 1][j] + A[i + 1][j]) / 2.;
        }
	}

    if (world_rank != world_size - 1) {
        MPI_Request req;
        MPI_Isend(A[size - 2], N, MPI_DOUBLE, world_rank + 1, 0, main_comm, &req);
        MPI_Request_free(&req);
    }
}

double update() {
    double local_eps = 0;
    if (world_start == world_end) {
        return local_eps; 
    }
	
    for(int i = 1; i < size - 1; i++) {
        for(int j = 1; j < N - 1; j++) {
		    double e = A[i][j];
		    A[i][j] = (A[i][j - 1] + A[i][j + 1]) / 2.;
		    local_eps = Max(local_eps, fabs(e - A[i][j]));
        }
	}

    if (world_rank != 0) {
        MPI_Request req;
        MPI_Isend(A[1], N, MPI_DOUBLE, world_rank - 1, 0, main_comm, &req);
        MPI_Request_free(&req);
    }
    if (world_rank + 1 != world_size) {
        MPI_Recv(A[size - 1], N, MPI_DOUBLE, world_rank + 1, 0, main_comm, MPI_STATUS_IGNORE);
    }

    return local_eps;
}


void verify(double **A, int size) { 
	double local_sum = 0.;
    double global_sum = 0.;
	for(int i = 1; i < size - 1; i++) {
	    for(int j = 0; j <= N - 1; j++) {
		    local_sum = local_sum + A[i][j] * (i + world_start) * (j + 1) / (N * N);
	    }
    }
    MPI_Reduce(&local_sum, &global_sum, 1, MPI_DOUBLE, MPI_SUM, 0, main_comm);
	
    if (world_rank == 0) {
        printf("  S = %f\n", global_sum);
    }
}
