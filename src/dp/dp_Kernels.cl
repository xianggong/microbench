__kernel void Calc_wg_offsets_wgf(
                            __global const uint* gHistArray,
                            __global uint* gPrefixsumArray,
                            uint bin_size
                            )
{
    uint lid = get_local_id(0);
    uint binId = get_group_id(0);

    uint group_offset = binId * bin_size;
    uint maxval = 0;

    do
    {
        uint binValue = gHistArray[group_offset + lid];
        uint prefix_sum = work_group_scan_exclusive_add( binValue );
        gPrefixsumArray[group_offset + lid] = prefix_sum + maxval;

        maxval += work_group_broadcast( prefix_sum + binValue, get_local_size(0)-1 );

        group_offset += get_local_size(0);
    }
    while(group_offset < (binId+1) * bin_size);
}
