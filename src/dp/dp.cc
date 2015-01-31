#include "dp.h"

#include <memory>

DynamicParallelism::DynamicParallelism()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);
}

DynamicParallelism::~DynamicParallelism()
{

}

void DynamicParallelism::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("dp_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0 -save-temps", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}



int main(int argc, char const *argv[])
{
	std::unique_ptr<DynamicParallelism> dp(new DynamicParallelism());
	
	return 0;
}