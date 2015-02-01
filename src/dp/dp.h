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
	int glbSize;
	int locSize;
	float factor;
	
	// User managed SVM buffers
	float *saxpy_src_0;
	float *saxpy_src_1;
	float *saxpy_dst_0;

	void init();
	void initKernel();
	void initBuffer();

	void clean();
	void cleanKernel();
	void cleanBuffer();

public:
	DynamicParallelism(int N = 8192);
	~DynamicParallelism();

	void runNaive();
	void runStride();
	void runDP();
	
};

#endif
