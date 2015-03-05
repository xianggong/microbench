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
