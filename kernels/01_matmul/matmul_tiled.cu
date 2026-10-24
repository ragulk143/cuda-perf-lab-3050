#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>
#define N 1024
#define TILE 16

__global__ void __launch_bounds__(256) matmul_tiled(__half* A, __half* B, __half* C, int n){
  __shared__ __half As[16][16];
  __shared__ __half Bs[16][16];
  int row = blockIdx.y * 16 + threadIdx.y;
  int col = blockIdx.x * 16 + threadIdx.x;
  float sum = 0;
  for(int t=0; t < n/16; ++t){
    As[threadIdx.y][threadIdx.x] = A[row*n + t*16 + threadIdx.x];
    Bs[threadIdx.y][threadIdx.x] = B[(t*16 + threadIdx.y)*n + col];
    __syncthreads();
    #pragma unroll
    for(int k=0;k<16;k++) sum += __half2float(As[threadIdx.y][k]) * __half2float(Bs[k][threadIdx.x]);
    __syncthreads();
  }
  C[row*n+col] = __float2half(sum);
}

int main(){
  size_t bytes = N*N*sizeof(__half);
  __half *h = (__half*)malloc(bytes);
  for(int i=0;i<N*N;i++) h[i] = __float2half(float(i%5));
  __half *dA,*dB,*dC; 
  cudaMalloc(&dA,bytes); cudaMalloc(&dB,bytes); cudaMalloc(&dC,bytes);
  cudaMemcpy(dA,h,bytes,cudaMemcpyHostToDevice);
  cudaMemcpy(dB,h,bytes,cudaMemcpyHostToDevice);
  dim3 block(16,16);
  dim3 grid(N/16, N/16);
  matmul_tiled<<<grid,block>>>(dA,dB,dC,N); cudaDeviceSynchronize();
  cudaEvent_t s,e; cudaEventCreate(&s); cudaEventCreate(&e);
  cudaEventRecord(s);
  for(int i=0;i<200;i++) matmul_tiled<<<grid,block>>>(dA,dB,dC,N);
  cudaEventRecord(e); cudaEventSynchronize(e);
  float ms; cudaEventElapsedTime(&ms,s,e);
  std::cout << "Tiled Avg: " << ms/200.0 << " ms" << std::endl;
  __half out[N*N]; cudaMemcpy(out,dC,bytes,cudaMemcpyDeviceToHost);
  std::cout << "check C[0]=" << __half2float(out[0]) << std::endl;
  return 0;
}
