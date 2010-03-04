#ifndef __CONST_H__
#define __CONST_H__

/* Matrix size */
#define MATSIZE 1024

/* Size of submatrices (block version only) 
 * 32 is the optimal value for 1024 X 1024 matrices for Cell processors*/
#define BLOCKSIZE 16

// Page size using HUGE_PAGES
#define HUGE_PAGE_SIZE 16*1024*1024

#endif /* __CONST_H__ */
