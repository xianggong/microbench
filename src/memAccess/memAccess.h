#ifndef MEM_ACCESS_H
#define MEM_ACCESS_H

#include <clUtil.h>

using namespace clHelper;

class memAccess
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
	memAccess();
	~memAccess();

	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();
	
};

#endif
