#include <cstdio>
#include <iostream>
#include <map>
#include <mpi.h>
#include <cuda_runtime.h>
#include <ATen/ATen.h>


std::map<MPI_Datatype, at::ScalarType> mpiToATDtypeMap = {
    {MPI_INT, at::kInt}, 
    {MPI_FLOAT, at::kFloat}
};


inline cudaError_t checkCuda(cudaError_t ret) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(ret));
        assert(ret == cudaSuccess);
    }
    return ret;
}


int main() {
    int worldSize, worldRank = -1;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int localRank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

    int numDevices = -1;
    checkCuda(cudaGetDeviceCount(&numDevices));
    checkCuda(cudaSetDevice(localRank % numDevices));

    int shape = 4;

    at::ScalarType dtype = mpiToATDtypeMap.at(MPI_FLOAT);
    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    at::Tensor tensorDevice = at::ones({shape, shape}, options) * localRank;

    auto tensorHost = tensorDevice.cpu();
    // output atomicity ignored
    printf("Global Rank: %d/%d, Local Rank: %d, Tensor:\n", worldRank, worldSize, localRank);
    std::cout << tensorHost << std::endl;

    auto tensorPtr = tensorHost.data_ptr();
    MPI_Allreduce(MPI_IN_PLACE, tensorPtr, shape * shape, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // output atomicity ignored
    printf("Global Rank: %d/%d, Local Rank: %d, AR Tensor:\n", worldRank, worldSize, localRank);
    std::cout << tensorHost << std::endl;

    tensorDevice = tensorHost.cuda();

    MPI_Finalize();
    return 0;
}