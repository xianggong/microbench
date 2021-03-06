__kernel void sm_reduce(         const int  numElems,
                        __global const int *src,
                        __global       int *dst)
{
    
}

__kernel void wgf_reduce(         const int  numElems,
                         __global const int *src,
                         __global       int *dst)
{
    uint global_id = get_global_id(0);

    // Reduce multiple elements per workitem
    int sum = 0;
    for (int i = global_id; i < numElems; i += get_global_size(0))
        sum += src[i];

    sum = work_group_reduce_add(sum);

    if (get_local_id(0) == 0)
        dst[get_group_id(0)] = sum;
}

__kernel void wgf_reduce_atomic(         const int  numElems,
                                __global const int *src,
                                __global       int *dst)
{
    uint global_id = get_global_id(0);

    // Reduce multiple elements per workitem
    int sum = 0;
    for (int i = global_id; i < numElems; i += get_global_size(0))
        sum += src[i];

    sum = work_group_reduce_add(sum);

    if (get_local_id(0) == 0)
        atomic_add(dst, sum);
}
