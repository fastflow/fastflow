float fmap3f(
	     __global float* R,
	     const uint useless,
	     const int i,
	     __global const float* A,
	     __global const float* sum_) {
    (void)useless; 
    const float sum = *sum_; 
    return R[i] + 1 / (A[i] + sum);
}

__kernel void kern_map3f(
			 __global float* input,
			 __global float* output,
			 const uint inSize,
			 const uint maxItems,
			 const uint offset,
			 const uint halo,
			 __global const float* env1,
			 __global const float* env2) {
    int i = get_global_id(0);
    int ig = i + offset;
    uint gridSize = get_local_size(0)*get_num_groups(0);
    while(i < maxItems)  {
	output[i+halo] = fmap3f(input+halo,inSize,ig,env1,env2);
	i += gridSize;
    }
}

