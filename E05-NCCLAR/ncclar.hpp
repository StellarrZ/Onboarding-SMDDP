#pragma once

#include <map>
#include <mpi.h>
#include <cuda_runtime.h>
#include <nccl.h>
#include <ATen/ATen.h>


std::map<MPI_Datatype, at::ScalarType> mpiToATDtypeMap = {
    {MPI_INT, at::kInt}, 
    {MPI_FLOAT, at::kFloat}
};


std::map<MPI_Datatype, ncclDataType_t> mpiToNCCLDtypeMap = {
    {MPI_INT, ncclInt}, 
    {MPI_FLOAT, ncclFloat}
};


// inline void checkCuda(cudaError_t ret);
// inline void checkNCCL(ncclResult_t ret);
// inline void checkMPI(int ret);