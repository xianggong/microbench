__kernel void pipe_producer(__global          float *src,
                            __write_only pipe float out_pipe)
{
    int gid = get_global_id(0);

    reserve_id_t res_id;
    res_id = reserve_write_pipe(out_pipe, 1);

    float src_pipe = src[gid] * 2.0;

    if(is_valid_reserve_id(res_id))
    {
        if(write_pipe(out_pipe, res_id, 0, &src_pipe) != 0)
            return;
        commit_write_pipe(out_pipe, res_id);
    }
}

__kernel void pipe_consumer(__global         float *dst,
                            __read_only pipe float in_pipe)
{
    int gid = get_global_id(0);

    reserve_id_t res_id;
    res_id = reserve_read_pipe(in_pipe, 1);

    float dst_pipe;

    if(is_valid_reserve_id(res_id))
    {
        if(read_pipe(in_pipe, res_id, 0, &dst_pipe) != 0)
            return;
        commit_read_pipe(in_pipe, res_id);
    }

    dst[gid] = dst_pipe * 3.0;
}