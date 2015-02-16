__kernel__ void offset(float* a, int s)
{
	int i = get_global_id(0) + s;
	a[i] = a[i] + 1;
}

__kernel__ void stride(float* a, int s)
{
	int i = (get_global_id(0)) * s;
	a[i] = a[i] + 1;
}
