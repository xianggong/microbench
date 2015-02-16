#include "memAccess.h"

#include <memory>

memAccess::memAccess()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);
}

memAccess::~memAccess()
{

}

void memAccess::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("memAccess_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}

void memAccess::InitBuffer()
{

}

void memAccess::FreeKernel()
{
	
}

void memAccess::FreeBuffer()
{
	
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<memAccess> tp(new memAccess());
	
	return 0;
}