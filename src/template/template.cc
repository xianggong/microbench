#include "template.h"

#include <memory>

Template::Template()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);
}

Template::~Template()
{

}

void Template::InitKernel()
{
	cl_int err;

	// Open kernel file
        file->open("template_Kernels.cl");

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
	std::unique_ptr<Template> tp(new Template());
	
	return 0;
}