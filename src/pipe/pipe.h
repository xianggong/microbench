#ifndef PIPI_H
#define PIPI_H

#include <clUtil.h>

using namespace clHelper;

class Pipe
{
    clRuntime *runtime;
    clFile    *file;

    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;

    cl_command_queue cmdQueue_0;
    cl_command_queue cmdQueue_1;

    cl_program       program;
    cl_kernel        kernel_0;
    cl_kernel        kernel_1;
    cl_kernel        kernel_pipe_producer;
    cl_kernel        kernel_pipe_consumer;

    float           *src;
    float           *dst;

    cl_mem           pipe;
    cl_uint          localSize;
    cl_uint          numPackets;
    cl_uint          packetSize;          

    /// Private functions
    void InitKernel();
    void InitBuffer();
    void InitPipe();

    void FreeKernel();
    void FreeBuffer();
    void FreePipe();

public:
    Pipe(int numPackets = 8192);
    ~Pipe();

    void Run();
    void RunPipe();
    
};

#endif
