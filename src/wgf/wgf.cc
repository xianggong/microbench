#include "wgf.h"

#include <math.h>
#include <memory>

#if ENABLE_PROFILE
#define clEnqueueNDRangeKernel clTimeNDRangeKernel
#endif

WorkGroupFunc::WorkGroupFunc(int N)
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0, true);

        numElems = N;
        numElemsBytes = numElems * sizeof(int);

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
        kernel_wgf_reduce_atomic = clCreateKernel(program, "wgf_reduce_atomic", &err);
        checkOpenCLErrors(err, "Failed to clCreateKernel wgf_reduce_atomic");
}

void WorkGroupFunc::InitBuffer()
{
        cl_int err;

        src_0 = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, numElemsBytes, 0);
        dst_0 = (int *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, numElemsBytes, 0);
        src_1 = (int *)clSVMAlloc(context, CL_MEM_READ_ONLY, numElemsBytes, 0);
        dst_1 = (int *)clSVMAlloc(context, CL_MEM_WRITE_ONLY, numElemsBytes, 0);

        int zero = 0;
        int one = 1;
        err  = clEnqueueSVMMemFill(cmdQueue, src_0, (const void *)&one, sizeof(int), numElemsBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, dst_0, (const void *)&zero, sizeof(int), numElemsBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, src_1, (const void *)&one, sizeof(int), numElemsBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, dst_1, (const void *)&zero, sizeof(int), numElemsBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to clEnqueueSVMMemFill");
}

void WorkGroupFunc::FreeKernel()
{
        cl_int err;

        err = clReleaseKernel(kernel_wgf_reduce);
        checkOpenCLErrors(err, "Failed to release kernel_wgf_reduce");
        err = clReleaseKernel(kernel_wgf_reduce_atomic);
        checkOpenCLErrors(err, "Failed to release kernel_wgf_reduce");

        err = clReleaseProgram(program);
        checkOpenCLErrors(err, "Failed to release program");
}

void WorkGroupFunc::FreeBuffer()
{
        clSVMFreeSafe(context, src_0);
        clSVMFreeSafe(context, dst_0);
        clSVMFreeSafe(context, src_1);
        clSVMFreeSafe(context, dst_1);
}

void WorkGroupFunc::RunSM()
{

}

void WorkGroupFunc::Run2Pass()
{
        cl_int err;

        size_t globalSize_0 = std::min(int(ceil(numElems/256) * 256), 1024);
        size_t localSize_0  = 256;
        size_t globalSize_1 = 256;
        size_t localSize_1  = 256;
        int N = numElems;

        err  = clSetKernelArg(kernel_wgf_reduce, 0, sizeof(int), (void *)&N);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 1, src_0);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 2, dst_0);
        checkOpenCLErrors(err, "Failed to set args in kernel_wgf_reduce");

        err = clTimeNDRangeKernel(
                cmdQueue,
                kernel_wgf_reduce,
                1,
                0, &globalSize_0, &localSize_0,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clTimeNDRangeKernel");

        int numWG = globalSize_0 / localSize_0;
        err  = clSetKernelArg(kernel_wgf_reduce, 0, sizeof(int), (void *)&numWG);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 1, dst_0);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce, 2, dst_0);
        checkOpenCLErrors(err, "Failed to set args in kernel_wgf_reduce");

        err = clTimeNDRangeKernel(
                cmdQueue,
                kernel_wgf_reduce,
                1,
                0, &globalSize_1, &localSize_1,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clTimeNDRangeKernel");     

        // Reduction result
        Dump(dst_0, 1);
}

void WorkGroupFunc::RunAtomic()
{
        cl_int err;

        size_t globalSize_0 = std::min(int(ceil(numElems/256) * 256), 1024);
        size_t localSize_0  = 256;
        int N = numElems;

        err  = clSetKernelArg(kernel_wgf_reduce_atomic, 0, sizeof(int), (void *)&N);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce_atomic, 1, src_1);
        err |= clSetKernelArgSVMPointer(kernel_wgf_reduce_atomic, 2, dst_1);
        checkOpenCLErrors(err, "Failed to set args in kernel_wgf_reduce");

        err = clTimeNDRangeKernel(
                cmdQueue,
                kernel_wgf_reduce_atomic,
                1,
                0, &globalSize_0, &localSize_0,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clTimeNDRangeKernel");

        // Reduction result
        Dump(dst_1, 1);
}

void WorkGroupFunc::Dump(int *svm_ptr, int N)
{
        cl_int err;

        clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, svm_ptr, sizeof(int) * N, 0, NULL, NULL);
        for (int i = 0; i < N; ++i)
                std::cout << dst_0[i] << std::endl;
        clEnqueueSVMUnmap(cmdQueue, svm_ptr, 0, NULL, NULL);
}

int main(int argc, char const *argv[])
{
        if (argc != 2)
        {
                printf("wgf #elements\n");
                exit(-1);
        }

        std::unique_ptr<WorkGroupFunc> wgf(new WorkGroupFunc(atoi(argv[1])));
        
        wgf->Run2Pass();
        wgf->RunAtomic();

        return 0;
}
