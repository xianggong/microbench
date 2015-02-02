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

__kernel void saxpy_dp(         const int    numElems,
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
    
    __global const float *src_0_child = src_0[child_offset];
    __global const float *src_1_child = src_0[child_offset];
    __global const float *dst_0_child = src_0[child_offset];

    queue_t defQ = get_default_queue();
    ndrange_t ndrange1 = ndrange_1D(child_global_sz);

    void (^saxpy_dp_child_wrapper)(void) = ^{saxpy_dp_child(child_global_sz, factor, src_0_child, src_1_child, dst_0_child);};
          
    int err_ret = enqueue_kernel(defQ, CLK_ENQUEUE_FLAGS_WAIT_KERNEL, ndrange1, saxpy_dp_child_wrapper);

}
