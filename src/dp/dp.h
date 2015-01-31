#ifndef DYNAMIC_PARALLELISM
#define DYNAMIC_PARALLELISM

#include <clUtil.h>

using namespace clHelper;

class DynamicParallelism
{
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue;

	cl_program       program;
	cl_kernel        kernel;

public:
	DynamicParallelism();
	~DynamicParallelism();

	void InitKernel();
	void InitBuffer();

	
};

#endif
