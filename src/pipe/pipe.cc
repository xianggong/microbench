#include "pipe.h"

#include <memory>

Pipe::Pipe(int numPackets)
        :localSize(64),
         numPackets(16384),
         packetSize(4)
{
        runtime  = clRuntime::getInstance();
        file     = clFile::getInstance();

        platform = runtime->getPlatformID();
        device   = runtime->getDevice();
        context  = runtime->getContext();

        cmdQueue_0 = runtime->getCmdQueue(0);
        cmdQueue_1 = runtime->getCmdQueue(1);

        this->numPackets = numPackets;

        // Init
        InitKernel();
        InitBuffer();
        InitPipe();
}

Pipe::~Pipe()
{
        FreeKernel();
        FreeBuffer();
        FreePipe();
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
        kernel_pipe_consumer = clCreateKernel(program, "pipe_consumer", &err);
        checkOpenCLErrors(err, "Failed to clCreateKernel pipe_consumer");
        kernel_pipe_producer = clCreateKernel(program, "pipe_producer", &err);
        checkOpenCLErrors(err, "Failed to clCreateKernel pipe_consumer");

}

void Pipe::InitBuffer()
{
        cl_int err;
        float one = 1.0f;

        size_t sizeBytes = packetSize * numPackets; 
        src = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeBytes, 0);
        dst = (float *)clSVMAlloc(context, CL_MEM_READ_WRITE | CL_MEM_SVM_FINE_GRAIN_BUFFER, sizeBytes, 0);
        if(!src || !dst)
        {
               std::cout << "Failed to allocate buffer" << std::endl;
               exit(-1);
        }

        err  = clEnqueueSVMMemFill(cmdQueue_0, src, (const void *)&one, sizeof(float), sizeBytes, 0, NULL, NULL);
        err  = clEnqueueSVMMemFill(cmdQueue_0, dst, (const void *)&one, sizeof(float), sizeBytes, 0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to fill SVM buffer");
        clFinish(cmdQueue_0);

        // Dump buffer
        printf("Before\n");
        for (int i = 0; i < numPackets; i += numPackets / 8)
                printf("%f %f\n", src[i], dst[i]);

}

void Pipe::InitPipe()
{
        cl_int err;
        pipe = clCreatePipe(context, CL_MEM_HOST_NO_ACCESS, packetSize, numPackets, NULL, &err);
        checkOpenCLErrors(err, "Failed to create pipe")
}

void Pipe::FreeKernel()
{
        cl_int err;

        err  = clReleaseKernel(kernel_pipe_consumer);
        err |= clReleaseKernel(kernel_pipe_producer);
        checkOpenCLErrors(err, "Failed to release kernel");

        clReleaseProgram(program);
        checkOpenCLErrors(err, "Failed to release program");
}

void Pipe::FreeBuffer()
{
        clSVMFreeSafe(context, src);
        clSVMFreeSafe(context, dst);
}

void Pipe::FreePipe()
{
        cl_int err;

        err = clReleaseMemObject(pipe);
        checkOpenCLErrors(err, "Failed to release pipe");
}

void Pipe::RunPipe()
{
        cl_int err;

        size_t globalWorkItems = numPackets;
        size_t localWorkItems  = localSize;

        // Producer
        err  = clSetKernelArgSVMPointer(kernel_pipe_producer, 0, src);
        err |= clSetKernelArg(kernel_pipe_producer, 1, sizeof(cl_mem), (void *)&pipe);
        checkOpenCLErrors(err, "Failed to clSetKernelArg");

        err  = clEnqueueNDRangeKernel(cmdQueue_0, 
                                      kernel_pipe_producer, 
                                      1, 
                                      NULL, 
                                      &globalWorkItems, 
                                      &localWorkItems, 
                                      0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to launch NDRange");

        // Consumer
        err  = clSetKernelArgSVMPointer(kernel_pipe_consumer, 0, dst);
        err |= clSetKernelArg(kernel_pipe_consumer, 1, sizeof(cl_mem), (void *)&pipe);
        checkOpenCLErrors(err, "Failed to clSetKernelArg");

        err  = clEnqueueNDRangeKernel(cmdQueue_1, 
                                      kernel_pipe_consumer, 
                                      1, 
                                      NULL, 
                                      &globalWorkItems, 
                                      &localWorkItems, 
                                      0, NULL, NULL);
        checkOpenCLErrors(err, "Failed to launch NDRange");

        // Dump buffer
        printf("After\n");
        for (int i = 0; i < numPackets; i += numPackets / 8)
                printf("%f %f\n", src[i], dst[i]);


}

int main(int argc, char const *argv[])
{
        if (argc != 2)
        {
                printf("pipe #elements\n");
                exit(-1);
        }

        std::unique_ptr<Pipe> pipe(new Pipe(atoi(argv[1])));
        
        pipe->RunPipe();

        return 0;
}
