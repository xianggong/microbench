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

	glbSize = 8192;
	locSize = 256;
	factor  = 2.3f;

	init();
}

DynamicParallelism::~DynamicParallelism()
{
	clean();
}

void DynamicParallelism::init()
{
	initKernel();
	initBuffer();
}

void DynamicParallelism::initKernel()
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
        checkOpenCLErrors(err, "Failed to create saxpy_stride kernel");
        kernel_saxpy_dp = clCreateKernel(program, "saxpy_dp", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_dp kernel");
}

void DynamicParallelism::initBuffer()
{
	cl_int err;
	size_t glbSizeBytes = glbSize * sizeof(float);

	saxpy_src_0 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
	saxpy_src_1 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
	saxpy_dst_0 = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, glbSizeBytes, 0);

	float two = 2.0f;
	float three = 3.0f;

	err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
	err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to fill SVM buffers");

	err = clFlush(cmdQueue);
	checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

	// Map SVM buffers 
	err  = clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MEM_READ_WRITE, saxpy_src_0, glbSizeBytes, 0, NULL, NULL);
	err |= clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MEM_READ_WRITE, saxpy_src_1, glbSizeBytes, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to map SVM buffers for initialization");

	// Initialize buffers
	for (int i = 0; i < glbSize; ++i)
		printf("%f %f\n", saxpy_src_0[i], saxpy_src_1[i]);

	// Unmap SVM buffers
	err  = clEnqueueSVMUnmap(cmdQueue, saxpy_src_0, 0, NULL, NULL);
	err |= clEnqueueSVMUnmap(cmdQueue, saxpy_src_1, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to unmap SVM buffers after initialization");

}

void DynamicParallelism::clean()
{
	cleanKernel();
	cleanBuffer();
}

void DynamicParallelism::cleanKernel()
{
	cl_int err;
	
	err = clReleaseKernel(kernel_saxpy_naive);
	err |= clReleaseKernel(kernel_saxpy_stride);
	err |= clReleaseKernel(kernel_saxpy_dp);
	checkOpenCLErrors(err, "Failed to release kernels");

	err = clReleaseProgram(program);
	checkOpenCLErrors(err, "Failed to release program");
}

void DynamicParallelism::cleanBuffer()
{
	clSVMFreeSafe(context, saxpy_src_0);
	clSVMFreeSafe(context, saxpy_src_1);
	clSVMFreeSafe(context, saxpy_dst_0);
}

void DynamicParallelism::runNaive()
{
	cl_int err;
	
	size_t globalSize = glbSize;
	size_t localSize  = locSize;

	err  = clSetKernelArg(kernel_saxpy_naive, 0, sizeof(int), (void *)&glbSize);
	err |= clSetKernelArg(kernel_saxpy_naive, 1, sizeof(int), (void *)&factor);
	err |= clSetKernelArgSVMPointer(kernel_saxpy_naive, 2, saxpy_src_0);
	err |= clSetKernelArgSVMPointer(kernel_saxpy_naive, 3, saxpy_src_1);
	err |= clSetKernelArgSVMPointer(kernel_saxpy_naive, 4, saxpy_dst_0);
	checkOpenCLErrors(err, "Failed to set args in saxpy_naive kernel");

        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_saxpy_naive,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

	// Map SVM buffers 
	size_t glbSizeBytes = glbSize * sizeof(float);	
	err  = clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MEM_READ_ONLY, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to map SVM buffers for checking result");

	// Check result
	for (int i = 0; i < glbSize / locSize; ++i)
		printf("%f\n", saxpy_dst_0[i]);

	// Unmap SVM buffers
	err  = clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);
	checkOpenCLErrors(err, "Failed to unmap SVM buffers after checking result");


}

void DynamicParallelism::runStride()
{

}

void DynamicParallelism::runDP()
{
	
}
int main(int argc, char const *argv[])
{
	std::unique_ptr<DynamicParallelism> dp(new DynamicParallelism());

	printf("Running...\n");

	dp->runNaive();
	dp->runStride();
	dp->runDP();
	
	printf("Done!\n");

	return 0;
}