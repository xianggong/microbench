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
	cl_kernel        kernel_wgf_reduce;

	static const int numElems = 65536;
	static const size_t numElemsBytes = numElems * sizeof(int);

	int *src_0;
	int *dst_0;

	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();

public:
	WorkGroupFunc();
	~WorkGroupFunc();

	void Run();
	
};

#endif
