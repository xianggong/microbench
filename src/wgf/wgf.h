#ifndef WORKGROUP_FUNC_H
#define WORKGROUP_FUNC_H

#include <clUtil.h>

using namespace clHelper;

class WorkGroupFunc
{
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue;

	cl_program       program;
	cl_kernel        kernel_sm_reduce;
	cl_kernel        kernel_wgf_reduce;
	cl_kernel        kernel_wgf_reduce_atomic;

	int numElems;
	size_t numElemsBytes;

	int *src_0;
	int *dst_0;
	int *src_1;
	int *dst_1;

	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();

public:
	explicit WorkGroupFunc(int N);
	~WorkGroupFunc();

	void RunSM();
	void Run2Pass();
	void RunAtomic();
	void Dump(int *svm_ptr, int numElems);
	
};

#endif
