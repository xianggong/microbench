#include "dp.h"

#include <sys/time.h>
#include <memory>

DynamicParallelism::DynamicParallelism(int N)
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);

        // Device side queue
        cmdQueueDev = runtime->getCmdQueue(1, CL_QUEUE_ON_DEVICE | CL_QUEUE_ON_DEVICE_DEFAULT);

        glbSize = N;
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

        // Create kernels
        kernel_saxpy_naive = clCreateKernel(program, "saxpy_naive", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_naive kernel");
        kernel_saxpy_stride = clCreateKernel(program, "saxpy_stride", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_stride kernel");
        kernel_saxpy_dp_no_wait = clCreateKernel(program, "saxpy_dp_no_wait", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_dp_no_wait kernel");
        kernel_saxpy_dp_wait_kernel = clCreateKernel(program, "saxpy_dp_wait_kernel", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_dp_wait_kernel kernel");
        kernel_saxpy_dp_wait_workgroup = clCreateKernel(program, "saxpy_dp_wait_workgroup", &err);
        checkOpenCLErrors(err, "Failed to create saxpy_dp_wait_workgroup kernel");
}

void DynamicParallelism::initBuffer()
{
        cl_int err;
        size_t glbSizeBytes = glbSize * sizeof(float);

        saxpy_src_0 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
        saxpy_src_1 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
        saxpy_dst_0 = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, glbSizeBytes, 0);
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
        err |= clReleaseKernel(kernel_saxpy_dp_no_wait);
        err |= clReleaseKernel(kernel_saxpy_dp_wait_kernel);
        err |= clReleaseKernel(kernel_saxpy_dp_wait_workgroup);
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
        float one = 1.0f;
        float two = 2.0f;
        float three = 3.0f;
        size_t glbSizeBytes = glbSize * sizeof(float);

        err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffers");

        err = clFlush(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
        
        size_t globalSize[1] = {(size_t)glbSize};
        size_t localSize[1]  = {(size_t)locSize};

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
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void DynamicParallelism::runStride()
{
        cl_int err;
        float one = 1.0f;
        float two = 2.0f;
        float three = 3.0f;
        size_t glbSizeBytes = glbSize * sizeof(float);

        err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffers");

        err = clFlush(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
        
        cl_int numCU;
        err = clGetDeviceInfo(device, CL_DEVICE_MAX_COMPUTE_UNITS, sizeof(cl_int), &numCU, NULL);
        checkOpenCLErrors(err, "Failed to query number of CU");

        size_t globalSize[1] = {(size_t) numCU * 64};
        size_t localSize[1]  = {(size_t) locSize};

        err  = clSetKernelArg(kernel_saxpy_stride, 0, sizeof(int), (void *)&glbSize);
        err |= clSetKernelArg(kernel_saxpy_stride, 1, sizeof(int), (void *)&factor);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_stride, 2, saxpy_src_0);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_stride, 3, saxpy_src_1);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_stride, 4, saxpy_dst_0);
        checkOpenCLErrors(err, "Failed to set args in saxpy_naive kernel");

        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_saxpy_stride,
                1,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");
}

void DynamicParallelism::runDPNoWait()
{
        cl_int err;
        float one = 1.0f;
        float two = 2.0f;
        float three = 3.0f;
        size_t glbSizeBytes = glbSize * sizeof(float);

        err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffers");

        err = clFlush(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
        
        size_t localSize[1]  = {(size_t)std::min(256, glbSize)};
        size_t globalSize[1] = {(size_t)glbSize / localSize[0]};

        err  = clSetKernelArg(kernel_saxpy_dp_no_wait, 0, sizeof(int), (void *)&glbSize);
        err |= clSetKernelArg(kernel_saxpy_dp_no_wait, 1, sizeof(int), (void *)&factor);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_no_wait, 2, saxpy_src_0);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_no_wait, 3, saxpy_src_1);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_no_wait, 4, saxpy_dst_0);
        checkOpenCLErrors(err, "Failed to set args in saxpy_naive kernel");

        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_saxpy_dp_no_wait,
                1,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

        // Dump buffer
        clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

        for (int i = 0; i < glbSize; i += glbSize / 8)
                printf("%f ", saxpy_dst_0[i]);
        printf("\n");

        clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

}

void DynamicParallelism::runDPWaitKernel()
{
        cl_int err;
        float one = 1.0f;
        float two = 2.0f;
        float three = 3.0f;
        size_t glbSizeBytes = glbSize * sizeof(float);

        err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffers");

        err = clFlush(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
        
        size_t localSize[1]  = {(size_t)std::min(256, glbSize)};
        size_t globalSize[1] = {(size_t)glbSize / localSize[0]};

        err  = clSetKernelArg(kernel_saxpy_dp_wait_kernel, 0, sizeof(int), (void *)&glbSize);
        err |= clSetKernelArg(kernel_saxpy_dp_wait_kernel, 1, sizeof(int), (void *)&factor);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_kernel, 2, saxpy_src_0);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_kernel, 3, saxpy_src_1);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_kernel, 4, saxpy_dst_0);
        checkOpenCLErrors(err, "Failed to set args in saxpy_naive kernel");

        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_saxpy_dp_wait_kernel,
                1,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

        // Dump buffer
        clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

        for (int i = 0; i < glbSize; i += glbSize / 8)
                printf("%f ", saxpy_dst_0[i]);
        printf("\n");

        clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

}

void DynamicParallelism::runDPWaitWorkgroup()
{
        cl_int err;
        float one = 1.0f;
        float two = 2.0f;
        float three = 3.0f;
        size_t glbSizeBytes = glbSize * sizeof(float);

        err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffers");

        err = clFlush(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
        
        size_t localSize[1]  = {(size_t)std::min(256, glbSize)};
        size_t globalSize[1] = {(size_t)glbSize / localSize[0]};

        err  = clSetKernelArg(kernel_saxpy_dp_wait_workgroup, 0, sizeof(int), (void *)&glbSize);
        err |= clSetKernelArg(kernel_saxpy_dp_wait_workgroup, 1, sizeof(int), (void *)&factor);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_workgroup, 2, saxpy_src_0);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_workgroup, 3, saxpy_src_1);
        err |= clSetKernelArgSVMPointer(kernel_saxpy_dp_wait_workgroup, 4, saxpy_dst_0);
        checkOpenCLErrors(err, "Failed to set args in saxpy_naive kernel");

        err = clEnqueueNDRangeKernel(
                cmdQueue,
                kernel_saxpy_dp_wait_workgroup,
                1,
                0, globalSize, localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clEnqueueNDRangeKernel");

        // Dump buffer
        clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

        for (int i = 0; i < glbSize; i += glbSize / 8)
                printf("%f ", saxpy_dst_0[i]);
        printf("\n");

        clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);

        err = clFinish(cmdQueue);
        checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

}

int main(int argc, char const *argv[])
{
        std::unique_ptr<DynamicParallelism> dp;

        if (argc > 2)
        {
                printf("Usage: ./dp #\n");
                return -1;
        }

        if (argc == 1)
                dp.reset(new DynamicParallelism(8192));
        else
                dp.reset(new DynamicParallelism(atoi(argv[1])));

        printf("Running...\n");

        dp->runNaive();
        dp->runStride();
        dp->runDPNoWait();
        dp->runDPWaitKernel();
        dp->runDPWaitWorkgroup();
        
        printf("Done!\n");

        return 0;
}