/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/*! 

 *  \file oclMacroes.hpp
 *  \ingroup 
 *
 *  \brief OpenCL map macroes
 *
 */

/* ***************************************************************************
 *
 *  This program is free software; you can redistribute it and/or modify it
 *  under the terms of the GNU Lesser General Public License version 3 as
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful, but WITHOUT
 *  ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 *  FITNESS FOR A PARTICULAR PURPOSE.  See the GNU Lesser General Public
 *  License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with this program; if not, write to the Free Software Foundation,
 *  Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 ****************************************************************************
 */



/* The following OpenCL code macros have been "inspired" from the 
 * SkePU OpenCL code 
 * http://www.ida.liu.se/~chrke/skepu/
 */

//  x=f(param)   'x' and 'param' have the same type
#define FFMAPFUNC(name, basictype, param, code)                         \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(" #basictype " " #param ") {\n" #code ";\n}\n"  \
"__kernel void kern_" #name "(\n"                                       \
"\t__global " #basictype "* input,\n"                                   \
"\t__global " #basictype "* output,\n"                                  \
"\tconst uint inSize,\n"                                                \
"\tconst uint maxItems) {\n"                                            \
"\t    int i = get_global_id(0);\n"                                     \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"          \
"\t    while(i < maxItems)  {\n"                                        \
"\t        output[i] = f" #name "(input[i]);\n"                         \
"\t        i += gridSize;\n"                                            \
"\t    }\n"                                                             \
"}"


//  x=f(param)   'x' and 'param' have different types
#define FFMAPFUNC2(name, outT, inT, param, code)                        \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #outT "|"                                                       \
#outT " f" #name "(" #inT " " #param ") {\n" #code ";\n}\n"             \
"__kernel void kern_" #name "(__global " #inT  "* input,\n"             \
"                             __global " #outT "* output,\n"            \
"                             const uint inSize,\n"                     \
"                             const uint maxItems) {\n"                 \
"           int i = get_global_id(0);\n"                                \
"           uint gridSize = get_local_size(0)*get_num_groups(0);\n"     \
"           while(i < maxItems)  {\n"                                   \
"              output[i] = f" #name "(input[i]);\n"                     \
"              i += gridSize;\n"                                        \
"           }\n"                                                        \
"}"

//  x=f(param,size, idx) 'param' is the input array, 
//                       'size' is the size of the input array,
//                       'param[idx]' and 'x' have the same type
#define FF_ARRAY(name, basictype, param, size, idx, code)               \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(\n"                                             \
"\t" #basictype "* " #param ",\n"                                       \
"\tconst uint " #size ",\n"                                             \
"\tconst int " #idx ") {\n"                                             \
"\t   "  #code ";\n"                                                    \
"}\n\b"                                                                 \
"__kernel void kern_" #name "(\n"                                       \
"\t__global " #basictype "* input,\n"                                   \
"\t__global " #basictype "* output,\n"                                  \
"\tconst uint inSize,\n"                                                \
"\tconst uint maxItems) {\n"                                            \
"\t    int i = get_global_id(0);\n"                                     \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"          \
"\t    while(i < maxItems)  {\n"                                        \
"\t        output[i] = f" #name "(input,inSize,i);\n"                   \
"\t        i += gridSize;\n"                                            \
"\t    }\n"                                                             \
"}"


//  x=f(param,size,idx,env) 'param' is the input array, 
//                          'size' is the size of the input array,
//                          'param[idx]' and 'x' have the same type
//                          'env' is constant
#define FF_ARRAY_CENV(name, basictype, param,size,idx,envT,env,code)    \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
"\n\n" #basictype " f" #name "(\n"                                      \
"\t__global " #basictype "* " #param ",const uint " #size ",\n"         \
"\tconst int " #idx ", __global const " #envT "* " #env ") {\n"         \
"\t   " #code ";\n"                                                     \
"}\n\n"                                                                 \
"__kernel void kern_" #name "(\n"                                       \
"\t__global " #basictype "* input,\n"                                   \
"\t__global " #basictype "* output,\n"                                  \
"\tconst uint inSize,\n"                                                \
"\t__global const " #envT "* env,\n"                                    \
"\tconst uint maxItems) {\n"                                            \
"\t    int i = get_global_id(0);\n"                                     \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"          \
"\t    while(i < maxItems)  {\n"                                        \
"\t        output[i] = f" #name "(input,inSize,i,env);\n"               \
"\t        i += gridSize;\n"                                            \
"\t    }\n"                                                             \
"}"

//  x=f(param,size,idx,env) 'param' is the input array, 
//                          'size' is the size of the input array,
//                          'param[idx]' and 'x' have different types
//                          'env' is constant
#define FF_ARRAY2_CENV(name, outT, inT,param,size,idx,envT,env,code)   \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #outT "|"                                                       \
"\n\n" #outT " f" #name "(\n"                                           \
"\t__global " #inT "* " #param ",const uint " #size ",\n"               \
"\tconst int " #idx ", __global const " #envT "* " #env ") {\n"         \
"\t   " #code ";\n"                                                     \
"}\n\n"                                                                 \
"__kernel void kern_" #name "(\n"                                       \
"\t__global " #inT  "* input,\n"                                        \
"\t__global " #outT "* output,\n"                                       \
"\tconst uint inSize,\n"                                                \
"\t__global const " #envT "* env,\n"                                    \
"\tconst uint maxItems) {\n"                                            \
"\t    int i = get_global_id(0);\n"                                     \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"          \
"\t    while(i < maxItems)  {\n"                                        \
"\t        output[i] = f" #name "(input,inSize,i,env);\n"               \
"\t        i += gridSize;\n"                                            \
"\t    }\n"                                                             \
"}"


//  x=f(param,size,idx,env1,env2)  'param' is the input array, 
//                                 'size' is the size of the input array,
//                                 'param[idx]' and 'x' have different types
//                                 'env1' and 'env2' are constants
#define FF_ARRAY2_CENV2(name,outT,inT,param,size,idx,env1T,env1,env2T,env2,code) \
    static char name[] =                                                         \
        "kern_" #name "|"                                                        \
        #outT "|"                                                                \
"\n\n" #outT " f" #name "(\n"                                                    \
"\t__global " #inT "* " #param ",const uint " #size ",\n"                        \
"\tconst int " #idx ",\n"                                                        \
"\t__global const " #env1T "* " #env1 ",\n"                                      \
"\t__global const " #env2T "* " #env2 ") {\n"                                    \
"\t   " #code ";\n"                                                              \
"}\n\n"                                                                          \
"__kernel void kern_" #name "(\n"                                                \
"\t__global " #inT  "* input,\n"                                                 \
"\t__global " #outT "* output,\n"                                                \
"\tconst uint inSize,\n"                                                         \
"\t__global const " #env1T "* env1,\n"                                           \
"\t__global const " #env2T "* env2,\n"                                           \
"\tconst uint maxItems) {\n"                                                     \
"\t    int i = get_global_id(0);\n"                                              \
"\t    uint gridSize = get_local_size(0)*get_num_groups(0);\n"                   \
"\t    while(i < maxItems)  {\n"                                                 \
"\t        output[i] = f" #name "(input,inSize,i,env1,env2);\n"                  \
"\t        i += gridSize;\n"                                                     \
"\t    }\n"                                                                      \
"}"


//  x=f(param1,param2)   'x', 'param1', 'param2' have the same type
#define FFREDUCEFUNC(name, basictype, param1, param2, code)             \
    static char name[] =                                                \
        "kern_" #name "|"                                               \
        #basictype "|"                                                  \
#basictype " f" #name "(" #basictype " " #param1 ",\n"                  \
                          #basictype " " #param2 ") {\n" #code ";\n}\n" \
"__kernel void kern_" #name "(__global " #basictype "* input, __global " #basictype "* output, const uint n, __local " #basictype "* sdata) {\n" \
"        uint blockSize = get_local_size(0);\n"                         \
"        uint tid = get_local_id(0);\n"                                 \
"        uint i = get_group_id(0)*blockSize + get_local_id(0);\n"       \
"        uint gridSize = blockSize*get_num_groups(0);\n"                \
"        float result = 0;\n"                                           \
"        if(i < n) { result = input[i]; i += gridSize; }\n"             \
"        while(i < n) {\n"                                              \
"          result = f" #name "(result, input[i]);\n"                    \
"          i += gridSize;\n"                                            \
"        }\n"                                                           \
"        sdata[tid] = result;\n"                                        \
"        barrier(CLK_LOCAL_MEM_FENCE);\n"                               \
"        if(blockSize >= 512) { if (tid < 256 && tid + 256 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 256]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 256) { if (tid < 128 && tid + 128 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid + 128]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >= 128) { if (tid <  64 && tid +  64 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  64]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  64) { if (tid <  32 && tid +  32 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  32]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  32) { if (tid <  16 && tid +  16 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +  16]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=  16) { if (tid <   8 && tid +   8 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   8]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   8) { if (tid <   4 && tid +   4 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   4]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   4) { if (tid <   2 && tid +   2 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   2]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(blockSize >=   2) { if (tid <   1 && tid +   1 < n) { sdata[tid] = f" #name "(sdata[tid], sdata[tid +   1]); } barrier(CLK_LOCAL_MEM_FENCE); }\n" \
"        if(tid == 0) output[get_group_id(0)] = sdata[tid];\n"          \
"}\n"


