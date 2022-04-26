#include<stdio.h>
#include<cstring>
#include<mpi.h>

#define ZERO 0
#define MSG_LEN 14

int main() {
    int size, rank = -1;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &size);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    char data[MSG_LEN] = {};
    char operation[9] = "received";
    if (rank == ZERO) {
        strcpy(data, "Hello World!\n");
        strcpy(operation, "sent");
    }
    MPI_Bcast(data, MSG_LEN, MPI_BYTE, ZERO, MPI_COMM_WORLD);
    printf("Process %d/%d %s message: %s\n", rank, size, operation, data);

    MPI_Finalize();
    return 0;
}