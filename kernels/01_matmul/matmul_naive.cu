#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>

#define N 1024

__global__ void matmul_naive(__half* A, __half* B, __half* C, int n){
  int row = blockIdx.y + blockDim.y * threadIdx.y;
  int col = blockIdx.x + blockDim.x * threadIdx.x;

  if(row < n && col < n){
    float sum = 0;
    for(int k=0;k<n;k++) sum += __half2float(A[row*n+k]) * __half2float(B[k*n+col]);
    C[row*n+col] = __float2half(sum);
  }
}

int main(){
  size_t bytes = N*N*sizeof(__half);
  __half *dA, *dB, *dC;
  cudaMalloc(&dA, bytes); cudaMalloc(&dB, bytes); cudaMalloc(&dC, bytes);
  dim3 block(16,16);
  dim3 grid((N+15)/16, (N+15)/16);
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for(int i=0;i<50;i++) matmul_naive<<<grid,block>>>(dA, dB, dC, N);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms, s, e);
  std::cout << "Avg: " << ms/50.0 << " ms" << std::endl;
  return 0;
}
