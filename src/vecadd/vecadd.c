#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <CL/opencl.h>
 
// OpenCL kernel. Each work item takes care of one element of c
const char *kernelSource =                                       "\n" \
"#pragma OPENCL EXTENSION cl_khr_fp64 : enable                    \n" \
"__kernel void vecAdd(  __global double *a,                       \n" \
"                       __global double *b,                       \n" \
"                       __global double *c,                       \n" \
"                       const unsigned int n)                    \n" \
"{                                                               \n" \
"    //Get our global thread ID                                  \n" \
"    int id = get_global_id(0);                                  \n" \
"                                                                \n" \
"    //Make sure we do not go out of bounds                      \n" \
"    if (id < n)                                                 \n" \
"        c[id] = a[id] + b[id];                                  \n" \
"}                                                               \n" \
                                                                "\n" ;
 
int main( int argc, char* argv[] )
{
    // Length of vectors
    unsigned int n = 100000;
 
    // Host input vectors
    double *h_a;
    double *h_b;
    // Host output vector
    double *h_c;
 
    cl_platform_id cpPlatform;        // OpenCL platform
    cl_device_id device_id;           // device ID
    cl_context context;               // context
    cl_command_queue queue;           // command queue
    cl_program program;               // program
    cl_kernel kernel;                 // kernel
 
    // Size, in bytes, of each vector
    size_t bytes = n*sizeof(double);
 
    size_t globalSize, localSize;
    cl_int err;
 
    // Number of work items in each local work group
    localSize = 64;
 
    // Number of total work items - localSize must be devisor
    globalSize = ceil(n/(float)localSize)*localSize;
 
    // Bind to platform
    err = clGetPlatformIDs(1, &cpPlatform, NULL);
 
    // Get ID for the device
    err = clGetDeviceIDs(cpPlatform, CL_DEVICE_TYPE_GPU, 1, &device_id, NULL);
 
    // Create a context  
    context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
 
    // Create a command queue 
    queue = clCreateCommandQueueWithProperties(context, device_id, 0, &err);
 
    // Create the compute program from the source buffer
    program = clCreateProgramWithSource(context, 1,
                            (const char **) & kernelSource, NULL, &err);
 
    // Build the program executable 
    clBuildProgram(program, 0, NULL, "-cl-std=CL2.0", NULL, NULL);
 
    // Create the compute kernel in the program we wish to run
    kernel = clCreateKernel(program, "vecAdd", &err);
 
     // Allocate memory for each vector on host
    h_a = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes, 0);
    h_b = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes, 0);
    h_c = (double*)clSVMAlloc(context, CL_MEM_READ_WRITE, bytes, 0);
 
    clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, h_a, bytes, 0, NULL, NULL);
    clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_WRITE, h_b, bytes, 0, NULL, NULL);

    // Initialize vectors on host
    int i;
    for( i = 0; i < n; i++ )
    {
        h_a[i] = sinf(i)*sinf(i);
        h_b[i] = cosf(i)*cosf(i);
    }

    clEnqueueSVMUnmap(queue, h_a, 0, NULL, NULL);
    clEnqueueSVMUnmap(queue, h_b, 0, NULL, NULL);

    clFinish(queue);

    // Set the arguments to our compute kernel
    err  = clSetKernelArgSVMPointer(kernel, 0, h_a);
    err |= clSetKernelArgSVMPointer(kernel, 1, h_b);
    err |= clSetKernelArgSVMPointer(kernel, 2, h_c);
    err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &n);
 
    // Execute the kernel over the entire range of the data set  
    err = clEnqueueNDRangeKernel(queue, kernel, 1, NULL, &globalSize, &localSize,
                                                              0, NULL, NULL);
 
    // Wait for the command queue to get serviced before reading back results
    clFinish(queue);
 
    // Read the results from the device
    clEnqueueSVMMap(queue, CL_TRUE, CL_MAP_READ, h_c, bytes, 0, NULL, NULL);
 
    //Sum up vector c and print result divided by n, this should equal 1 within error
    double sum = 0;
    for(i=0; i<n; i++)
        sum += h_c[i];
    printf("final result: %f\n", sum/n);

    clEnqueueSVMUnmap(queue, h_c, 0, NULL, NULL);
 
    // release OpenCL resources
    clReleaseProgram(program);
    clReleaseKernel(kernel);
    clReleaseCommandQueue(queue);
    clReleaseContext(context);
 
    //release host memory
    clSVMFree(context, h_a);
    clSVMFree(context, h_b);
    clSVMFree(context, h_c);
 
    return 0;
}