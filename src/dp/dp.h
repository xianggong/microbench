#ifndef DYNAMIC_PARALLELISM_H
#define DYNAMIC_PARALLELISM_H

#include <clUtil.h>

using namespace clHelper;

class DynamicParallelism
{
	// Helper
	clRuntime *runtime;
	clFile    *file;

	// Runtimes, auto clean
	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue;

	// User managed OpenCL objects
	cl_program       program;
	cl_kernel        kernel_saxpy_naive;
	cl_kernel        kernel_saxpy_stride;
	cl_kernel        kernel_saxpy_dp;

	// Parameters
	static const int glbSize = 8192;
	static const int blkSize = 512;

	// User managed SVM buffers
	float *saxpy_src_0;
	float *saxpy_src_1;
	float *saxpy_dst_0;

	void InitKernel();
	void InitBuffer();

	void CleanKernel();
	void CleanBuffer();

public:
	DynamicParallelism();
	~DynamicParallelism();

	void runNaive();
	void runStride();
	void runDP();
	
};

#endif
