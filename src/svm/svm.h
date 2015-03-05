#ifndef SVM_H
#define SVM_H

#include <clUtil.h>

using namespace clHelper;

class SVM
{
    clRuntime        *runtime;
    clFile           *file;

    cl_platform_id   platform;
    cl_device_id     device;
    cl_context       context;
    cl_command_queue cmdQueue;

    cl_program       program;
    cl_kernel        kernel_saxpy_naive;

    // Parameters
    int              glbSize;
    int              locSize;
    float            factor;

    float           *saxpy_src_0;
    float           *saxpy_src_1;
    float           *saxpy_dst_0;

public:
    SVM(int N = 8192);
    ~SVM();

    void InitKernel();
    void InitBuffer();

    void FreeKernel();
    void FreeBuffer();
    
    void RunCoarseGrain();
    void RunFineGrainBuffer();
    void RunFineGrainSystem();

};

#endif
