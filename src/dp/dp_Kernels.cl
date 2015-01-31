__kernel void saxpy_naive(         const int    numElems,
                                   const float  factor,
                          __global const float *src_0,
                          __global const float *src_1,
                          __global       float *dst_0)
{
    uint gid = get_global_id(0);

    if (gid < N)
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

__kernel void saxpy_dp(         const int    numElems,
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
    
}