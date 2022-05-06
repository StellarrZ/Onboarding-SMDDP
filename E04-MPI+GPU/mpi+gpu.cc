#include <cstdio>
#include <iostream>
#include <map>
#include <string>
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


int main() {
    int worldSize, worldRank = -1;
    MPI_Init(NULL, NULL);
    MPI_Comm_size(MPI_COMM_WORLD, &worldSize);
    MPI_Comm_rank(MPI_COMM_WORLD, &worldRank);

    int localRank = atoi(std::getenv("OMPI_COMM_WORLD_LOCAL_RANK"));

    int numDevices = -1;
    checkCuda(cudaGetDeviceCount(&numDevices));
    checkCuda(cudaSetDevice(localRank % numDevices));

    const int shape = 4;

    at::ScalarType dtype = mpiToATDtypeMap.at(MPI_FLOAT);
    auto options = at::TensorOptions().dtype(dtype).device(at::kCUDA);
    at::Tensor tensorDevice = at::ones({shape, shape}, options) * localRank;

    auto tensorHost = tensorDevice.cpu();
    // output atomicity ignored
    std::cout << "Global Rank: " << worldRank << "/" << worldSize << ", Local Rank: " << localRank;
    std::cout << ", Tensor:\n" << tensorHost << std::endl;

    auto tensorPtr = tensorHost.data_ptr();
    MPI_Allreduce(MPI_IN_PLACE, tensorPtr, shape * shape, MPI_FLOAT, MPI_SUM, MPI_COMM_WORLD);
    // output atomicity ignored
    std::cout << "Global Rank: " << worldRank << "/" << worldSize << ", Local Rank: " << localRank;
    std::cout << ", Tensor:\n" << tensorHost << std::endl;

    tensorDevice = tensorHost.cuda();

    const std::string testResult = test_sum(tensorDevice, dtype, shape, worldSize);
    // output atomicity ignored
    std::cout << "Global Rank: " << worldRank << "/" << worldSize << " Test " << testResult << std::endl;

    MPI_Finalize();
    return 0;
}