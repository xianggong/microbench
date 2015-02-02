__kernel void wgf_reduce(__global const int *src,
			 __global       int *dst)
{
	uint gid = get_global_id(0);
	uint lid = get_local_id(0);
	dst[lid] = work_group_reduce_add(src[gid]);
}