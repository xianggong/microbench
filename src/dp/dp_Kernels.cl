__kernel void saxpy_naive(         const int    numElems,
                                   const float  factor,
                          __global const float *src_0,
                          __global const float *src_1,
                          __global       float *dst_0)
{
    uint gid = get_global_id(0);

    if (gid < numElems)
        dst_0[gid] = factor * src_0[gid] * src_1[gid];
}

__kernel void saxpy_stride(         const int    numElems,
                                    const float  factor,
                           __global const float *src_0,
                           __global const float *src_1,
                           __global       float *dst_0)
{
    uint gid = get_global_id(0);
    uint step = get_global_size(0);

    for (int i = gid; i < numElems; i += step)
        dst_0[gid] = factor * src_0[gid] * src_1[gid];
}

void saxpy_dp_child(         const int    numElems,
                             const float  factor,
                    __global const float *src_0,
                    __global const float *src_1,
                    __global       float *dst_0)
{
    uint gid = get_global_id(0);

    if (gid < numElems)
        dst_0[gid] = factor * src_0[gid] * src_1[gid];
}

__kernel void saxpy_dp_no_wait(         const int    numElems,
                                        const float  factor,
                               __global const float *src_0,
                               __global const float *src_1,
                               __global       float *dst_0)
{
    // Each workitem launches a child ndrange
    // Total # of workitems = numElems
    uint global_id = get_global_id(0);
    uint global_sz = get_global_size(0);

    uint child_global_sz = numElems / global_sz;
    uint child_offset = global_id * child_global_sz;
    
    __global const float *src_0_child = &src_0[child_offset];
    __global const float *src_1_child = &src_1[child_offset];
    __global       float *dst_0_child = &dst_0[child_offset];

    // Default queue need to be created in host side before launching kernels
    queue_t defQ = get_default_queue();
    ndrange_t ndrange = ndrange_1D(child_global_sz);

    void (^saxpy_dp_child_wrapper)(void) = ^{saxpy_dp_child(child_global_sz, factor, src_0_child, src_1_child, dst_0_child);};
          
    int err_ret = enqueue_kernel(defQ, CLK_ENQUEUE_FLAGS_NO_WAIT, ndrange, saxpy_dp_child_wrapper);

}

__kernel void saxpy_dp_wait_kernel(         const int    numElems,
                                            const float  factor,
                                   __global const float *src_0,
                                   __global const float *src_1,
                                   __global       float *dst_0)
{
    // Each workitem launches a child ndrange
    // Total # of workitems = numElems
    uint global_id = get_global_id(0);
    uint global_sz = get_global_size(0);

    uint child_global_sz = numElems / global_sz;
    uint child_offset = global_id * child_global_sz;
    
    __global const float *src_0_child = &src_0[child_offset];
    __global const float *src_1_child = &src_1[child_offset];
    __global       float *dst_0_child = &dst_0[child_offset];

    // Default queue need to be created in host side before launching kernels
    queue_t defQ = get_default_queue();
    ndrange_t ndrange = ndrange_1D(child_global_sz);

    void (^saxpy_dp_child_wrapper)(void) = ^{saxpy_dp_child(child_global_sz, factor, src_0_child, src_1_child, dst_0_child);};
          
    int err_ret = enqueue_kernel(defQ, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange, saxpy_dp_child_wrapper);

}

__kernel void saxpy_dp_wait_workgroup(         const int    numElems,
                                               const float  factor,
                                      __global const float *src_0,
                                      __global const float *src_1,
                                      __global       float *dst_0)
{
    // Each workitem launches a child ndrange
    // Total # of workitems = numElems
    uint global_id = get_global_id(0);
    uint global_sz = get_global_size(0);

    uint child_global_sz = numElems / global_sz;
    uint child_offset = global_id * child_global_sz;
    
    __global const float *src_0_child = &src_0[child_offset];
    __global const float *src_1_child = &src_1[child_offset];
    __global       float *dst_0_child = &dst_0[child_offset];

    // Default queue need to be created in host side before launching kernels
    queue_t defQ = get_default_queue();
    ndrange_t ndrange = ndrange_1D(child_global_sz);

    void (^saxpy_dp_child_wrapper)(void) = ^{saxpy_dp_child(child_global_sz, factor, src_0_child, src_1_child, dst_0_child);};
          
    int err_ret = enqueue_kernel(defQ, CLK_ENQUEUE_FLAGS_WAIT_WORK_GROUP, ndrange, saxpy_dp_child_wrapper);

}
