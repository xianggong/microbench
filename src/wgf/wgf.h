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
	cl_kernel        kernel;

public:
	WorkGroupFunc();
	~WorkGroupFunc();

	void InitKernel();
	void InitBuffer();

	
};

#endif
