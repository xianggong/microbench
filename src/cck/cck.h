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

	std::vector<cl_command_queue> queues;

	cl_program       program;
	std::vector<cl_kernel> kernels;

	cl_command_queue getCmdQueue(int N) { 
		if (N < queues.size())
			return queues[N];
		return nullptr;
	}

	cl_kernel getKernel(int N) {
		if (N < kernels.size())
			return kernels[N];
		return nullptr;
	}

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
	CCK();
	~CCK();

	void RunSingle();
	void RunMulti();
	
};

#endif
