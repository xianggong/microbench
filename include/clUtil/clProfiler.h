#ifndef CL_PROFILER_H
#define CL_PROFILER_H

#include <memory>

namespace clHelper
{

// Enqueue and profile a kernel
cl_int clProfileNDRangeKernel(cl_command_queue cmdQ,
                              cl_kernel        kernel,
                              cl_uint          wd,
                              const size_t *   glbOs,
                              const size_t *   glbSz,
                              const size_t *   locSz,
                              cl_uint          numEvt,
                              const cl_event * evtLst,
                              cl_event *       evt)
{
        cl_int   err;
        cl_int   enqueueErr;
        cl_event perfEvent;
        cl_command_queue_properties cmdQProp;

        // Enable profiling of command queue
        // err = clSetCommandQueueProperty(cmdQ, CL_QUEUE_PROFILING_ENABLE, true, NULL);
        // checkOpenCLErrors(err, "Failed to enable profiling on command queue");

        // Enqueue kernel
        enqueueErr = clEnqueueNDRangeKernel(cmdQ, kernel, wd, glbOs, glbSz, locSz, 0, NULL, &perfEvent);
        checkOpenCLErrors(enqueueErr, "Failed to profile on kernel");
        clWaitForEvents(1, &perfEvent);
        
        // Get profiling information
        cl_ulong start = 0, end = 0;
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_START, sizeof(cl_ulong), &start, NULL);
        clGetEventProfilingInfo(perfEvent, CL_PROFILING_COMMAND_END, sizeof(cl_ulong), &end, NULL);
        cl_double execTimeMs = (cl_double)(end - start)*(cl_double)(1e-06); 

        // Get kernel name
        char kernelName[1024];
        err = clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 1024 * sizeof(char), (void *)kernelName, NULL);

        // printf
        printf("Kernel %s costs %f ms\n", kernelName, execTimeMs);

        return enqueueErr;
}

}

#endif
