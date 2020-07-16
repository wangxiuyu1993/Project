#include "include/cuda_runtime.h"
#include "include/device_launch_parameters.h"

#include <stdio.h>
#include <iostream>

int main()
{
	cudaError_t cudaStatus;
	int nNum = 0;
	cudaDeviceProp cudaProp;

   // 获得可用GPU个数
	cudaStatus = cudaGetDeviceCount(&nNum);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaGetDeviceCount failed!");
		return 1;
	}
	for (int i = 0; i<nNum; i++)
	{
	    // 获得属性
		cudaGetDeviceProperties(&cudaProp, i);
	}

	for (int j=0;j<nNum; ++j)
	{
		printf(" %d, name: %s, totalGlobalMem: %d (GB), sharedMemPerBlock: %d (KB), warpSize: %d, multiProcessorCount: %d, compute capability: %d.%d \n",
			j,cudaProp.name,cudaProp.totalGlobalMem/1024/1024/1024, cudaProp.sharedMemPerBlock/1024, cudaProp.warpSize, 
			cudaProp.multiProcessorCount, cudaProp.major, cudaProp.minor);
	}

	getchar();
	return 0;
}
