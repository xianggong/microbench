__kernel void wgf_reduce(__global const int *src,
			 __global       int *dst)
{
	uint gid = get_global_id(0);
	dst[gid] = work_group_reduce(src[gid]);
}