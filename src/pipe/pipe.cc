#include "pipe.h"

#include <memory>

Pipe::Pipe()
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();

        cmdQueue = runtime->getCmdQueue(0);

        // Init
        InitKernel();
        InitBuffer();
}

Pipe::~Pipe()
{
        FreeBuffer();
        FreeKernel();
}

void Pipe::InitKernel()
{
        cl_int err;

        // Open kernel file
        file->open("pipe_Kernels.cl");

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
        int numCU = runtime->getNumComputeUnit();
        cl_kernel kernel;
        for (int i = 0; i < numCU; ++i)
        {
                kernel = clCreateKernel(program, "Pipe_dummy", &err);
                checkOpenCLErrors(err, "Failed to clCreateKernel Pipe_dummy");
                kernels.push_back(kernel);
        }

        printf("Init kernel done\n");
}

void Pipe::InitBuffer()
{
        cl_int err;
        float one = 1.0f;

        size_t sizeBytes = sizeof(float) * numElems;
        srcDst = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE, sizeBytes, 0);
        if(!srcDst)
        {
               std::cout << "Failed to allocate buffer" << std::endl;
               exit(-1);
        }

        err  = clEnqueueSVMMemFill(getCmdQueue(0), srcDst, (const void *)&one, sizeof(float), sizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffer");
        printf("Init buffer done\n");
}

void Pipe::FreeKernel()
{
        cl_int err;

        for( auto &kernel : kernels)
        {
                err = clReleaseKernel(kernel);
                checkOpenCLErrors(err, "Failed to release kernel");
        }

        clReleaseProgram(program);
        checkOpenCLErrors(err, "Failed to release program");
        printf("Free kernel done\n");
}

void Pipe::FreeBuffer()
{
        clSVMFreeSafe(context, srcDst);
        printf("Free buffer done\n");
}

void Pipe::RunSingle()
{
        std::cout << "RunSingle" << std::endl;
        Dump(srcDst, numElems);
        cl_int err;

        // A single kernel handles all data
        size_t globalSize = runtime->getNumComputeUnit() * 64;
        size_t localSize  = 64;
        int N = numElems;

        cl_kernel krnl = getKernel(0);

        err  = clSetKernelArg(krnl, 0, sizeof(int), (void *)&N);
        err |= clSetKernelArgSVMPointer(krnl, 1, srcDst);
        checkOpenCLErrors(err, "Failed to set args in kernel 0");

        err = clTimeNDRangeKernel(
                getCmdQueue(0),
                krnl,
                1,
                0, &globalSize, &localSize,
                0, 0, 0
        );
        checkOpenCLErrors(err, "Failed at clTimeNDRangeKernel");

        Dump(srcDst, numElems);
}

void Pipe::RunMulti()
{
        std::cout << "RunMulti" << std::endl;
        cl_int err;

        // Launching multiple kernels to process the data
        size_t globalSize = 64;
        size_t localSize  = 64;
        int N = numElems;

        cl_kernel krnl;
        for (int i = 0; i < kernels.size(); ++i)
        {
                krnl = getKernel(i);
                int srcDstOff = i * 64;

                err  = clSetKernelArg(krnl, 0, sizeof(int), (void *)&N);
                err |= clSetKernelArgSVMPointer(krnl, 1, &srcDst[srcDstOff]);
                checkOpenCLErrors(err, "Failed to set args in kernel");

                err = clTimeNDRangeKernel(
                        getCmdQueue(i),
                        krnl,
                        1,
                        0, &globalSize, &localSize,
                        0, 0, 0
                );
                checkOpenCLErrors(err, "Failed at clTimeNDRangeKernel");
        }
        
        Dump(srcDst, numElems);

}

void Pipe::Dump(float *svm_ptr, int numElems)
{
        cl_int err;

        clEnqueueSVMMap(getCmdQueue(0), CL_TRUE, CL_MAP_READ, svm_ptr, sizeof(float) * numElems, 0, NULL, NULL);
        for (int i = 0; i < numElems; ++i)
                std::cout << svm_ptr[i] << std::endl;
        clEnqueueSVMUnmap(getCmdQueue(0), svm_ptr, 0, NULL, NULL);

}

int main(int argc, char const *argv[])
{
        std::unique_ptr<Pipe> tp(new Pipe());
        
        tp->RunSingle();
        tp->RunMulti();

        return 0;
}
