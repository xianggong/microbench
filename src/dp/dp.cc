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
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
        checkOpenCLErrors(err, "Failed to build program...\n");

        // Create kernels
        kernel_saxpy_naive = clCreateKernel(program, "saxpy_naive", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_naive kernel");
        kernel_saxpy_stride = clCreateKernel(program, "saxpy_stride", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_naive kernel");
        kernel_saxpy_dp = clCreateKernel(program, "saxpy_dp", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_naive kernel");
}

void DynamicParallelism::InitBuffer()
{
	cl_int err;
	size_t glbSizeBytes = glbSize * sizeof(float);

	saxpy_src_0 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
	saxpy_src_1 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
	saxpy_dst_0 = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, glbSizeBytes, 0);

	// Map SVM buffers for 
	err  = clEnqueueSVMMap(cmdQueue, true, CL_MEM_READ_WRITE, saxpy_src_0, glbSizeBytes, 0, NULL, NULL);
	err |= clEnqueueSVMMap(cmdQueue, true, CL_MEM_READ_WRITE, saxpy_src_1, glbSizeBytes, 0, NULL, NULL);
	err |= clEnqueueSVMMap(cmdQueue, true, CL_MEM_READ_WRITE, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to map SVM buffers for initialization");

	err  = clEnqueueSVMUnmap(cmdQueue, saxpy_src_0, 0, NULL, NULL);
	err |= clEnqueueSVMUnmap(cmdQueue, saxpy_src_1, 0, NULL, NULL);
	err |= clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to unmap SVM buffers after initialization");

}

void DynamicParallelism::CleanKernel()
{
	cl_int err;
	
	err = clReleaseKernel(kernel_saxpy_naive);
	err |= clReleaseKernel(kernel_saxpy_stride);
	err |= clReleaseKernel(kernel_saxpy_dp);
	checkOpenCLErrors(err, "Failed to release kernels");

	err = clReleaseProgram(program);
	checkOpenCLErrors(err, "Failed to release program");
}

void DynamicParallelism::CleanBuffer()
{
	clSVMFreeSafe(context, saxpy_src_0);
	clSVMFreeSafe(context, saxpy_src_1);
	clSVMFreeSafe(context, saxpy_dst_0);
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<DynamicParallelism> dp(new DynamicParallelism());
	
	return 0;
}