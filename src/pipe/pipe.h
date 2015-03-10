#ifndef PIPI_H
#define PIPI_H

#include <clUtil.h>

using namespace clHelper;

class Pipe
{
	clRuntime *runtime;
	clFile    *file;

	cl_platform_id   platform;
	cl_device_id     device;
	cl_context       context;

	cl_command_queue cmdQueue;

	cl_program       program;
	cl_kernel        kernel_0;
	cl_kernel        kernel_1;
	cl_kernel        kernel_pipe_producer;
	cl_kernel        kernel_pipe_consumer;

	int              numElems;
	float           *srcDst;

	/// Private functions
	void InitKernel();
	void InitBuffer();

	void FreeKernel();
	void FreeBuffer();

	/// Dump buffer data for debug
	void Dump(float *svm_ptr, int numElems);

public:
	Pipe();
	~Pipe();

	void Run();
	void RunPipe();
	
};

#endif
