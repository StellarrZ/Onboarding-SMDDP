#include "ncclar.hpp"

#include <iostream>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <ATen/ATen.h>


inline void checkCuda(cudaError_t ret) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(ret));
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
}


inline void checkNCCL(ncclResult_t ret) {
    if (ret != ncclSuccess) {
        fprintf(stderr, "NCCL Runtime Error: %s\n", ncclGetErrorString(ret));
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
}


inline void checkMPI(int ret) {
    if (ret != MPI_SUCCESS) {
        fprintf(stderr, "MPI Error Type: %d\n", ret);
        MPI_Abort(MPI_COMM_WORLD, 1); 
    }
}


inline std::string test_sum(at::Tensor &tensorDevice, const at::ScalarType dtype, const int shape, const int size) {
    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    auto answer = at::ones({shape, shape}, options) * ((size - 1) * size / 2);

    std::string testResult;
    if (answer.equal(tensorDevice)) {
        testResult = "Passed.";
    }
    else {
        testResult = "Failed.";
    }

    return testResult;
}


int main(int argc, char** argv) {
    const int root = 0;

    int worldSize, worldRank = -1;
    checkMPI(MPI_Init(NULL, NULL));
    checkMPI(MPI_Comm_size(MPI_COMM_WORLD, &worldSize));
    checkMPI(MPI_Comm_rank(MPI_COMM_WORLD, &worldRank));

    int localRank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

    int numDevices = -1;
    checkCuda(cudaGetDeviceCount(&numDevices));
    checkCuda(cudaSetDevice(localRank % numDevices));   // Only support worldSize <= numDevices in this circumstance

    ncclUniqueId commId;
    if (worldRank == root) {
        checkNCCL(ncclGetUniqueId(&commId));
    }
    checkMPI(MPI_Bcast(&commId, sizeof(commId), MPI_BYTE, root, MPI_COMM_WORLD));
    ncclComm_t comm;
    checkNCCL(ncclCommInitRank(&comm, worldSize, commId, worldRank));

    const int shape = 4;

    at::ScalarType dtype = mpiToATDtypeMap.at(MPI_FLOAT);
    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    at::Tensor tensorDevice = at::ones({shape, shape}, options) * localRank;
    auto tensorPtr = tensorDevice.data_ptr();

    cudaStream_t stream;
    checkCuda(cudaStreamCreate(&stream));
    checkNCCL(ncclAllReduce(tensorPtr, tensorPtr, shape * shape, mpiToNCCLDtypeMap.at(MPI_FLOAT), ncclSum, comm, stream));

    const std::string testResult = test_sum(tensorDevice, dtype, shape, worldSize);
    std::cout << "Global Rank: " << worldRank << "/" << worldSize << " Test " << testResult << std::endl;

    checkNCCL(ncclCommDestroy(comm));
    checkMPI(MPI_Finalize());
    return 0;
}