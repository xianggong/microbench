#include "wgf.h"

#include <sys/time.h>
#include <memory>

double time_stamp()
{
        struct timeval t;
        if(gettimeofday(&t, 0) != 0)
        	exit(-1);
        return t.tv_sec + t.tv_usec/1e6;
}

WorkGroupFunc::WorkGroupFunc()
{
	runtime  = clRuntime::getInstance();
	file     = clFile::getInstance();

	platform = runtime->getPlatformID();
	device   = runtime->getDevice();
	context  = runtime->getContext();
	cmdQueue = runtime->getCmdQueue(0);

	InitKernel();
	InitBuffer();
}

WorkGroupFunc::~WorkGroupFunc()
{
	FreeBuffer();
	FreeKernel();
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
        err = clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
        if (err != CL_SUCCESS)
        {
                // Debug
                char buf[0x10000];
                clGetProgramBuildInfo( program,
                                        device,
                                        CL_PROGRAM_BUILD_LOG,
                                        0x10000,
                                        buf,
                                        NULL);
                printf("%s\n", buf);
                exit(-1);
        }

        // Create kernel
        kernel_wgf_reduce = clCreateKernel(program, "wgf_reduce", &err);
        checkOpenCLErrors(err, "Failed to clCreateKernel wgf_reduce");
}

void WorkGroupFunc::InitBuffer()
{
	cl_int err;

	src_0 = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, numElemsBytes, 0);
	dst_0 = (int *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, numElemsBytes, 0);

        int one = 1;
        err  = clEnqueueSVMMemFill(cmdQueue, src_0, (const void *)&one, sizeof(int), numElemsBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, dst_0, (const void *)&one, sizeof(int), numElemsBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMMemFill src_0");
}

void WorkGroupFunc::FreeKernel()
{
	cl_int err;

	err = clReleaseKernel(kernel_wgf_reduce);
	checkOpenCLErrors(err, "Failed to release kernel_wgf_reduce");

	err = clReleaseProgram(program);
	checkOpenCLErrors(err, "Failed to release program");
}

void WorkGroupFunc::FreeBuffer()
{
	clSVMFreeSafe(context, src_0);
	clSVMFreeSafe(context, dst_0);
}

void WorkGroupFunc::Run()
{
	cl_int err;

	size_t globalSize_0 = std::min((numElems + 255)/256, 1024);
	size_t localSize_0  = 256;
        size_t globalSize_1 = 256;
        size_t localSize_1  = 256;
        int N = numElems;

        err  = clSetKernelArg(kernel_wgf_reduce, 0, sizeof(int), (void *)&N);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 1, src_0);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 2, dst_0);
        checkOpenCLErrors(err, "Failed to set args in kernel_wgf_reduce");

        double start = time_stamp();
        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_wgf_reduce,
                1,
                0, &globalSize_0, &localSize_0,
                0, 0, 0
        );
        double end = time_stamp();
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
        printf("Pass 0 takes %f\n", end - start);

        err  = clSetKernelArg(kernel_wgf_reduce, 0, sizeof(int), (void *)&N);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 1, dst_0);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 2, dst_0);
        checkOpenCLErrors(err, "Failed to set args in kernel_wgf_reduce");

        start = time_stamp();
        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_wgf_reduce,
                1,
                0, &globalSize_1, &localSize_1,
                0, 0, 0
        );
        end = time_stamp();
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");     
        printf("Pass 1 takes %f\n", end - start);

}

void WorkGroupFunc::Dump()
{
	cl_int err;

	clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, dst_0, numElemsBytes, 0, NULL, NULL);
	for (int i = 0; i < numElems; ++i)
		printf("%d\n", dst_0[i]);
}

int main(int argc, char const *argv[])
{
	std::unique_ptr<WorkGroupFunc> wgf(new WorkGroupFunc());
	
	wgf->Run();
	wgf->Dump();

	return 0;
}
