#include "svm.h"

#include <memory>

SVM::SVM(int N)
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();
        cmdQueue = runtime->getCmdQueue(0);

        // cl_int err = 0;
        
        // // Get platform
        // err = clGetPlatformIDs(1, &platform, NULL);
        // checkOpenCLErrors(err, "Failed at clGetPlatformIDs");

        // // Get ID for the device
        // err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
        // checkOpenCLErrors(err, "Failed at clGetDeviceIDs");

        // // Create a context
        // context = clCreateContext(0, 1, &device, NULL, NULL, &err);
        // checkOpenCLErrors(err, "Failed at clCreateContext");

        // // Create a cmdQueue
        // cmdQueue = clCreateCommandQueueWithProperties(context, device, NULL, &err);
        // checkOpenCLErrors(err, "Failed at clCreateContext");

        glbSize = N;
        locSize = 256;
        factor  = 2.3f;

        InitKernel();
}

SVM::~SVM()
{
        FreeKernel();

        // cl_int err;

        // err = clReleaseCommandQueue(cmdQueue);
        // checkOpenCLErrors(err, "Failed at clReleaseCommandQueue");

        // err = clReleaseContext(context);
        // checkOpenCLErrors(err, "Failed at clReleaseContext");

        // err = clReleaseDevice(device);
        // checkOpenCLErrors(err, "Failed at clReleaseDevice");
        // printf("~SVM\n");
}

void SVM::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("svm_Kernels.cl");

        // Create program
        const char *source = file->getSourceChar();
        program = clCreateProgramWithSource(context, 1, 
                (const char **)&source, NULL, &err);
        checkOpenCLErrors(err, "Failed to create Program with source\n");

        // Create program with OpenCL 2.0 support
        err = clBuildProgram(program, 0, NULL, "-I. -cl-std=CL2.0", NULL, NULL);
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
}

void SVM::FreeKernel()
{
        cl_int err;
        
        err = clReleaseKernel(kernel_saxpy_naive);
        checkOpenCLErrors(err, "Failed to release kernels");

        err = clReleaseProgram(program);
        checkOpenCLErrors(err, "Failed to release program");    

        // printf("FreeKernel\n");
}

void SVM::RunCoarseGrain()
{
        if (runtime->isSVMavail(SVM_COARSE))
        {
                cl_int err;

                // Init Buffer
                size_t glbSizeBytes = glbSize * sizeof(float);

                saxpy_src_0 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
                saxpy_src_1 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY, glbSizeBytes, 0);
                saxpy_dst_0 = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, glbSizeBytes, 0);

                // FIXME: It seems AMD's driver is buggy on command queue

                // Fill Buffer
                // float one = 1.0f;
                // float two = 2.0f;
                // float three = 3.0f;

                // err  = clEnqueueSVMMemFill(cmdQueue, saxpy_src_0, (const void *)&one, sizeof(float), glbSizeBytes, 0, NULL, NULL);
                // err |= clEnqueueSVMMemFill(cmdQueue, saxpy_src_1, (const void *)&two, sizeof(float), glbSizeBytes, 0, NULL, NULL);
                // err |= clEnqueueSVMMemFill(cmdQueue, saxpy_dst_0, (const void *)&three, sizeof(float), glbSizeBytes, 0, NULL, NULL);
                // checkOpenCLErrors(err, "Failed to fill SVM buffers");

                // err = clFlush(cmdQueue);
                // checkOpenCLErrors(err, "Failed to clFlush cmdQueue");

                clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, saxpy_src_0, glbSizeBytes, 0, NULL, NULL);
                clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, saxpy_src_1, glbSizeBytes, 0, NULL, NULL);
                clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_WRITE, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMMap\n");

                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_0[i] = 1.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_1[i] = 2.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_dst_0[i] = 3.0f;

                clEnqueueSVMUnmap(cmdQueue, saxpy_src_0, 0, NULL, NULL);
                clEnqueueSVMUnmap(cmdQueue, saxpy_src_1, 0, NULL, NULL);
                clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMUnmap\n");

                // Run kernel        
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
                // printf("clEnqueueNDRangeKernel\n");

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMMap\n");

                // Dump buffer
                clEnqueueSVMMap(cmdQueue, CL_TRUE, CL_MAP_READ, saxpy_dst_0, glbSizeBytes, 0, NULL, NULL);

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMUnmap\n");

                // for (int i = 0; i < glbSize; i += glbSize / 8)
                //         printf("%f ", saxpy_dst_0[i]);
                // printf("\n");

                clEnqueueSVMUnmap(cmdQueue, saxpy_dst_0, 0, NULL, NULL);

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMMap\n");

                // Free buffer
                clSVMFree(context, saxpy_src_0);
                clSVMFree(context, saxpy_src_1);
                clSVMFree(context, saxpy_dst_0);
                // printf("clSVMFree\n");
        }
}

void SVM::RunFineGrainBuffer()
{
        if (runtime->isSVMavail(SVM_FINE))
        {
                cl_int err;

                // Init Buffer
                size_t glbSizeBytes = glbSize * sizeof(float);

                // Fine grain still needs to use clSVMAlloc
                saxpy_src_0 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER, glbSizeBytes, 0);
                saxpy_src_1 = (float *)clSVMAlloc(context, CL_MEM_READ_ONLY | CL_MEM_SVM_FINE_GRAIN_BUFFER, glbSizeBytes, 0);
                saxpy_dst_0 = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, glbSizeBytes, 0);

                // No need to explicit map/unmap
                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_0[i] = 1.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_1[i] = 2.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_dst_0[i] = 3.0f;

                // Run kernel        
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
                // printf("clEnqueueNDRangeKernel\n");

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMMap\n");

                // Dump buffer
                // for (int i = 0; i < glbSize; i += glbSize / 8)
                //         printf("%f ", saxpy_dst_0[i]);
                // printf("\n");

                // Free buffer
                clSVMFree(context, saxpy_src_0);
                clSVMFree(context, saxpy_src_1);
                clSVMFree(context, saxpy_dst_0);
                // printf("clSVMFree\n");
        }

}

void SVM::RunFineGrainSystem()
{
        if (runtime->isSVMavail(SVM_SYSTEM))
        {
                cl_int err;

                // Init Buffer
                size_t glbSizeBytes = glbSize * sizeof(float);

                // Fine grain system alloc
                saxpy_src_0 = (float *)aligned_alloc(sizeof(float), glbSizeBytes);
                saxpy_src_1 = (float *)aligned_alloc(sizeof(float), glbSizeBytes);
                saxpy_dst_0 = (float *)aligned_alloc(sizeof(float), glbSizeBytes);

                // No need to explicit map/unmap
                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_0[i] = 1.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_src_1[i] = 2.0f;
                for (int i = 0; i < glbSize; ++i)
                        saxpy_dst_0[i] = 3.0f;

                // Run kernel        
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
                // printf("clEnqueueNDRangeKernel\n");

                err = clFinish(cmdQueue);
                checkOpenCLErrors(err, "Failed to clFlush cmdQueue");
                // printf("clFinish SVMMap\n");

                // Dump buffer
                for (int i = 0; i < glbSize; i += glbSize / 8)
                        printf("%f ", saxpy_dst_0[i]);
                printf("\n");

                // Free buffer
                clSVMFree(context, saxpy_src_0);
                clSVMFree(context, saxpy_src_1);
                clSVMFree(context, saxpy_dst_0);
                // printf("clSVMFree\n");
        }

}

int main(int argc, char const *argv[])
{
        std::unique_ptr<SVM> svm;
        if (argc != 2)
        {
                std::cout << "Usage: ./svm #" << std::endl;
                return -1;
        }

        if (argc == 1)
                svm.reset(new SVM(8192));
        else
                svm.reset(new SVM(atoi(argv[1])));

        svm->RunCoarseGrain();
        svm->RunFineGrainBuffer();
        svm->RunFineGrainSystem();

        return 0;
}