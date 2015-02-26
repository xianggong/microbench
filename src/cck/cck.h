#ifndef CCK_H
#define CCK_H

#include <clUtil.h>

using namespace clHelper;

class CCK
{
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;
	cl_command_queue cmdQueue0;
	cl_command_queue cmdQueue1;
	cl_command_queue cmdQueue2;
	cl_command_queue cmdQueue3;
	cl_command_queue cmdQueue4;
	cl_command_queue cmdQueue5;
	cl_command_queue cmdQueue6;
	cl_command_queue cmdQueue7;
	cl_command_queue cmdQueue8;
	cl_command_queue cmdQueue9;
	cl_command_queue cmdQueue10;
	cl_command_queue cmdQueue11;
	cl_command_queue cmdQueue12;
	cl_command_queue cmdQueue13;
	cl_command_queue cmdQueue14;
	cl_command_queue cmdQueue15;

	cl_program       program;
	cl_kernel        kernel;

public:
	CCK();
	~CCK();

	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();
	
};

#endif
