__kernel void cck_dummy(         const int    numElems,
                          __global const float *srcDst)
{
    uint gid = get_global_id(0);

    for (int i = 0; i < 128; ++i)
    {
            if (gid < numElems)
                srcDst[gid] *= 1.2f;
                srcDst[gid] += srcDst[gid];
    }
}
