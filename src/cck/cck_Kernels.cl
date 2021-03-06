__kernel void cck_dummy(            const int    numElems,
                          __global        float *srcDst)
{
    uint gid = get_global_id(0);

    if (gid < numElems)
    for (int i = 0; i < 65536; ++i)
    {
        srcDst[gid] *= 1.0f;
        srcDst[gid] += 1.0f;
        srcDst[gid] /= 1.0f;
    }
}
