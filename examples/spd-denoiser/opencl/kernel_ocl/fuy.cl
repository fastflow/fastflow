#define MAX(a,b) (((a) > (b)) ? (a) : (b))
#define MIN(a,b) (((a) < (b)) ? (a) : (b))
#define _ABS(a)	   (((a) < 0) ? -(a) : (a))

__kernel void fuy (__global uchar *res,
		   __global const uchar *im,
		   __global const uint *noisy,
		   __global const int *noisymap,
		   __global uchar *diff,
		   __global float *groupwise_residuals,
		   __local float *local_residuals,
		   const uint workgroup_size,
		   const uint n_noisy,
		   const uint h,
		   const uint w,
		   const float alfa,
		   const float beta)
{
  const unsigned int noisy_idx = get_global_id(0);
  const int local_idx = get_local_id(0);
  local_residuals[local_idx] = 0;
  if(noisy_idx < n_noisy) {
    const uint idx = noisy[noisy_idx];
    //get the pixel and the four closest neighbors (with replication-padding)
    uchar pixel = im[idx];
    //up
    //int idx_neighbor = (idx >= w) ? (idx - w) : idx;
    int idx_neighbor = idx - w * (idx >= w);
    uchar up_val = im[idx_neighbor];
    uchar up_noisy = (noisymap[idx_neighbor] >= 0);
    //down
    //idx_neighbor = (idx < ((h -1) * w)) ? (idx + w) : idx;
    idx_neighbor =  idx + w * (idx < ((h -1) * w));
    uchar down_val = im[idx_neighbor];
    uchar down_noisy = (noisymap[idx_neighbor] >= 0);
    //left
    //idx_neighbor = ((idx % w) > 0) ? idx - 1 : idx;
    idx_neighbor = idx - ((idx % w) > 0);
    uchar left_val = im[idx_neighbor];
    uchar left_noisy = (noisymap[idx_neighbor] >= 0);
    //right
    //idx_neighbor = ((idx % w) < (w - 1)) ? idx + 1 : idx;
    idx_neighbor =  idx + ((idx % w) < (w - 1));
    uchar right_val = im[idx_neighbor];
    uchar right_noisy = (noisymap[idx_neighbor] >= 0);

    //compute the correction
    uchar u = 0, new_val = 0;
    float S;
    float Fu, Fu_prec = FLT_MAX;
    float beta_ = beta / 2;
    for(int uu=0; uu<256; ++uu) {
      u = (uchar) uu;
      Fu = 0.0f;
      S = 0.0f;
      S += (float)(2-up_noisy) * native_powr(alfa, _ABS(uu - (int)up_val));
      S += (float)(2-down_noisy) * native_powr(alfa, _ABS(uu - (int)down_val));
      S += (float)(2-left_noisy) * native_powr(alfa, _ABS(uu - (int)left_val));
      S += (float)(2-right_noisy) * native_powr(alfa, _ABS(uu - (int)right_val));
      Fu += _ABS((float)u - (float)pixel) + (beta_) * S;
      if(Fu < Fu_prec) {
      	new_val = u;
	Fu_prec = Fu;
      }
    }

    //update res
    res[idx] = new_val;

    //thread-wise diff and residual
    uchar new_diff = (uchar)_ABS((int)new_val - (int)noisymap[idx]);
    float new_residual = _ABS((float)diff[noisy_idx] - (float)new_diff);
    diff[noisy_idx] = new_diff;

    //workgroup-wise reduction
    local_residuals[local_idx] = new_residual;
    barrier(CLK_LOCAL_MEM_FENCE);
    for(int s=workgroup_size/2; s>0; s/=2) {
      if (local_idx < s)
	local_residuals[local_idx] += local_residuals[local_idx + s];
      barrier(CLK_LOCAL_MEM_FENCE);
    }

    //store workgroup residual
    if (local_idx == 0)
      groupwise_residuals[get_group_id(0)] = local_residuals[0];
  }
}





__kernel void init (__global uchar *diff,
		    __global uchar *residual,
		    const uint n_noisy)
{
  const unsigned int noisy_idx = get_global_id(0);
  if(noisy_idx < n_noisy) {
    diff[noisy_idx] = 0;
  }
  if(get_local_id(0) == 0)
    residual[get_group_id(0)] = 0;
}
