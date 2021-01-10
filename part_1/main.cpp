#include <iostream>
#include <vector>
#include <utility>
#include <cstdlib>
#include <ctime>
#include <mpi.h>

#define N 5
#define Rec() std::cout << next_x << " " << next_y << " " << "recived message from " << prev_x << " " << prev_y << " " << tag << std::endl;
#define Sen() std::cout << prev_x << " " << prev_y << " " << "sended message to " <<  next_x << " " << next_y << " " << tag << std::endl;
#define Wat() std::cout << next_x << " " << next_y << " " << "waiting message from " << prev_x << " " << prev_y << " " << tag << std::endl;
using Path = std::vector<std::pair<int, int>>;

int world_rank;
int world_size;

void get_position(int &x, int &y) {
    x = world_rank / N;
    y = world_rank % N;
}

int get_index(int x, int y) {
    return N * y + x;
}

int main(int an, char **as) {
    std::srand(std::time(0));
    MPI_Init(NULL, NULL);
    
    MPI_Status status;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int x, y;
    uint32_t L = 100, K = 10;
    uint32_t num = L / (2 * K);
    double *message = (double *) malloc(L*sizeof(double));

    get_position(x, y);
    if (x == 0 and y == 0) {
        for (int i = 0; i < L; i++) {
            message[i] = rand() % 1000 + 1000.0 / (rand() % 1000);
            std::cout << message[i] << " ";
        }
        std::cout << std::endl;
    }

    //std::cout << x << " " << y << " index: " << get_index(x, y) << std::endl;
    MPI_Barrier(MPI_COMM_WORLD);
    std::vector<Path> paths{
        {
            {0,0}, {1, 0}, {4, 4} //{0, 2}, {0, 3}, {0, 4}, {1, 4}, {2, 4}, {3, 4}, {4, 4}
        },
       // {
       //     {0,0}, {1, 0}, {4, 4} //{2, 0}, {3, 0}, {4, 0}, {4, 1}, {4, 2}, {4, 3}, {4, 4}
       // }
    };

    std::vector<std::pair<int, int>> messages_tags = {{0, 1215}};//, {1, 1216}};
    MPI_Request request;


    for (auto&& [path_index, tag] : messages_tags) {
        auto prev_x = 0;
        auto prev_y = 0;
        for (auto&& [next_x, next_y] : paths[path_index]) {
            if (next_x == 0 and next_y == 0) {
                continue;
            }
            if (next_x == x and next_y == y) {
                uint32_t c = 0;
                if (x == 4 and y == 4 and path_index) {
                    c = K * num;
                }
                Wat();
                //for (uint32_t k = 0; k < K; k++) {
                    std::cout << x << " " << y << " " << " revc c: " << c << " " << get_index(prev_x, prev_y) << std::endl;
                    MPI_Recv(message + c, 1, MPI_DOUBLE, 
                            get_index(prev_x, prev_y), tag + get_index(prev_x, prev_y), MPI_COMM_WORLD, &status);
                //}
                Rec();
                break;
            }
            prev_x = next_x;
            prev_y = next_y;
        }
    }
    int32_t buff_size = L*sizeof(double) + MPI_BSEND_OVERHEAD;
    double* buff = (double *) malloc(buff_size);
    MPI_Buffer_attach(buff, buff_size);

    for (auto&& [path_index, tag] : messages_tags) {
        auto prev_x = 0;
        auto prev_y = 0;
        for (auto&& [next_x, next_y] : paths[path_index]) {
            //if (next_x == 4 and next_y == 4)
            //    continue;
            if (next_x == 0 and next_y == 0) {
                continue;
            }

            if (prev_x == x and prev_y == y) {
                uint32_t c = 0;
                if (x == 0 and y == 0 and path_index) {
                    c = K * num;
                }
                
                //for (uint32_t k = 0; k < K; k++) {
                
                    std::cout << x << " " << y  << " " << " send c: " << c << " " << get_index(next_x, next_y) << std::endl;
                    MPI_Bsend(message + c, 1, MPI_DOUBLE, 
                            get_index(next_x, next_y), tag + get_index(next_x, next_y), MPI_COMM_WORLD);
                //}
                Sen();
                break;
            }
            prev_x = next_x;
            prev_y = next_y;
        }
    }
    MPI_Buffer_detach(buff, &buff_size);
    free(buff);
    MPI_Barrier(MPI_COMM_WORLD);

    if (x == 4 and y == 4) {
        for (int i = 0; i < L; i++) {
            std::cout << message[i] << " ";
        }
        std::cout << std::endl;
    }

    MPI_Barrier(MPI_COMM_WORLD);
    free(message);
    MPI_Finalize();
    return 0;
}
