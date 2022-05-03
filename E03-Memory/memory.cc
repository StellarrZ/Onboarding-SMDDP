#include <cstdlib>
#include <cstdio>
#include <assert.h>
#include <time.h>
// #include <nvToolsExt.h>
// #include <cuda.h>
#include <cuda_runtime.h>


inline cudaError_t checkCuda(cudaError_t ret) {
    if (ret != cudaSuccess) {
        fprintf(stderr, "CUDA Runtime Error: %s\n", cudaGetErrorString(ret));
        assert(ret == cudaSuccess);
    }
    return ret;
}


inline bool checkElements(char *hostBuffer, const char initVal, const int bufferLen) {
    for (int i = 0; i < bufferLen; ++i) {
        if (hostBuffer[i] ^ initVal) {
            return false;
        }
    }
    return true;
}


int main() {
    const char initVal = 0;
    size_t bufferLen = (1 << 30) * sizeof(char);
    void *hostBuffer = nullptr;
    void *deviceBuffer = nullptr;

    checkCuda(cudaSetDevice(0));
    checkCuda(cudaMalloc(&deviceBuffer, bufferLen));

    checkCuda(cudaMemset(deviceBuffer, initVal, bufferLen));

    hostBuffer = malloc(bufferLen);

    clock_t start, end = 0;
    start = clock();
    cudaMemcpyAsync(hostBuffer, deviceBuffer, bufferLen, cudaMemcpyDeviceToHost);
    end = clock();
    printf("\nDtoH Throughput: %.3f GB/s\n\n", (double)(bufferLen >> 30) * CLOCKS_PER_SEC / (double)(end - start));

    assert(checkElements((char *)hostBuffer, initVal, bufferLen) == true);
    
    return 0;
}