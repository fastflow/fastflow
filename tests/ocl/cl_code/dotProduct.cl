float fmapf(float a, float b) {
return (a * b);;
}
__kernel void kern_mapf(
	__global float* input,
	__global float* output,
	const uint inSize,
	const uint maxItems,
	const uint offset,
	const uint halo,
	__global const float* env) {
	    int i = get_global_id(0);
	    uint gridSize = get_local_size(0)*get_num_groups(0);
	    while(i < maxItems)  {
	        output[i] = fmapf(input[i], env[i]);
	        i += gridSize;
	    }
}
float freducef(float x,
float y) {
return (x+y);;
}
__kernel void kern_reducef(__global float* input, const uint halo, __global float* output, const uint n, __local float* sdata, float idElem) {
        uint blockSize = get_local_size(0);
        uint tid = get_local_id(0);
        uint i = get_group_id(0)*blockSize + get_local_id(0);
        uint gridSize = blockSize*get_num_groups(0);
        float result = idElem; input += halo;
        if(i < n) { result = input[i]; i += gridSize; }
        while(i < n) {
          result = freducef(result, input[i]);
          i += gridSize;
        }
        sdata[tid] = result;
        barrier(CLK_LOCAL_MEM_FENCE);
        if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = freducef(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }
        if(tid == 0) output[get_group_id(0)] = sdata[tid];
}
