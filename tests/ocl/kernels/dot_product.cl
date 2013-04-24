// Simple compute kernel which computes the dot product of two input arrays 
//
										
__kernel void dotProduct(                                                
   __global float* input1,
   __global float* input2,                                            
   __global float* output)/*,                                          
     const  uint2  inputDimensions)*/                                      
{        
    //uint width  = inputDimensions.x;
    //uint height = inputDimensions.y;
    //int cnt = width*height;                                                           
    int i = get_global_id(0);                                         
    //if(i < cnt)                                                  
       output[i] = input1[i] * input2[i];                              
}                                                                      ;
 
////////////////////////////////////////////////////////////////////////////////