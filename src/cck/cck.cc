#include "cck.h"

#include <memory>

CCK::CCK()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();

	cmdQueue0 = runtime->getCmdQueue(0);
	cmdQueue1 = runtime->getCmdQueue(1);
	cmdQueue1 = runtime->getCmdQueue(2);
	cmdQueue1 = runtime->getCmdQueue(3);
	cmdQueue1 = runtime->getCmdQueue(4);
	cmdQueue1 = runtime->getCmdQueue(5);
	cmdQueue1 = runtime->getCmdQueue(6);
	cmdQueue1 = runtime->getCmdQueue(7);
	cmdQueue1 = runtime->getCmdQueue(8);
	cmdQueue1 = runtime->getCmdQueue(9);
	cmdQueue1 = runtime->getCmdQueue(10);
	cmdQueue1 = runtime->getCmdQueue(11);
	cmdQueue1 = runtime->getCmdQueue(12);
	cmdQueue1 = runtime->getCmdQueue(13);
	cmdQueue1 = runtime->getCmdQueue(14);
	cmdQueue1 = runtime->getCmdQueue(15);
}

CCK::~CCK()
{

}

void CCK::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("cck_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}

void CCK::InitBuffer()
{

}

void CCK::FreeKernel()
{
	
}

void CCK::FreeBuffer()
{
	
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<CCK> tp(new CCK());
	
	return 0;
}