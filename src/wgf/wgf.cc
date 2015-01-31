#include "wgf.h"

#include <memory>

WorkGroupFunc::WorkGroupFunc()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);
}

WorkGroupFunc::~WorkGroupFunc()
{

}

void WorkGroupFunc::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("wgf_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");
}



int main(int argc, char const *argv[])
{
	std::unique_ptr<WorkGroupFunc> wgf(new WorkGroupFunc());
	
	return 0;
}