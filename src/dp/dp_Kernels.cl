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

    dst_0[gid] = 5.0;
}

__kernel void saxpy_dp(         const int    numElems,
                                const float  factor,
                       __global const float *src_0,
                       __global const float *src_1,
                       __global       float *dst_0)
{
    uint gid = get_global_id(0);
    uint gsz = get_global_size(0);
    uint lid = get_local_id(0);
    uint lsz = get_local_size(0);

    if (lid == 0)
    {
      queue_t defQ = get_default_queue();
      ndrange_t ndrange1 = ndrange_1D(lsz);    

      void (^saxpy_dp_child_wrapper)(void) = ^{saxpy_dp_child(get_local_size, factor, src_0, src_1, dst_0);};
            
      int err_ret = enqueue_kernel(defQ,CLK_ENQUEUE_FLAGS_WAIT_KERNEL,ndrange1,saxpy_dp_child_wrapper);      
    }


}
