/****************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*
 * Cholesky matrix factorization algorithm, block version.
 *
 * Authors: Marco Fais       (faism@users.sourceforge.net)
 *          Alessio Pascucci (pascucci@di.unipi.it)
 *
 */


#include <vector>
#include <iostream>
#include <cmath>
#include <fcntl.h>
//#include <sys/mman.h>
#include <ff/farm.hpp>
#include <cholconst.h>
#include <complex.h>
#include <common.h>


using namespace ff;

/* Support fuctions */

/* Cholesky factorization of a block: l = Factor(a[bi][bj])
 *
 * param a	pointer to source matrix (block matrix)
 * param bi	row index of the block to compute in matrix a
 * param bj	column index of the block to compute in matrix a
 * param l	pointer to destination block (ptr to block, not ptr to block matrix)
 */
void factor(const comp_t *a, unsigned bi, unsigned bj, comp_t *l)
{
	unsigned i, j, k;	// indexes used in loops
	float sumSQR;		// support variable
	comp_t sum;			// support variable
	unsigned offseti = bi * BLOCKSIZE;	// row offset for matrix a
	unsigned offsetj = bj * BLOCKSIZE;	// column offset for matrix a
	
	// Begin factorization
	for (j = 0; j < BLOCKSIZE; j++) {
		sumSQR = 0.;
		
		for (k = 0; k < j; k++) {
			sumSQR += (l[j * BLOCKSIZE + k].real * l[j * BLOCKSIZE + k].real) +
			(l[j * BLOCKSIZE + k].imag * l[j * BLOCKSIZE + k].imag);
		}
		
		// L[j][j]^2
		sumSQR = a[(offseti + j) * MATSIZE + offsetj + j].real - sumSQR;
		l[j * BLOCKSIZE + j].real = sqrt(sumSQR);
		
		for (i = j + 1; i < BLOCKSIZE; i++) {
			sum.real = 0.;
			sum.imag = 0.;
			
			for (k = 0; k < j; k++) {
				comp_t tmp;
				comp_t trasp;
				trasp.real = l[j * BLOCKSIZE + k].real;
				trasp.imag = -(l[j * BLOCKSIZE + k].imag);
				COMP_MUL(l[i * BLOCKSIZE + k], trasp, tmp);
				COMP_SUM(sum, tmp, sum);
				// sum += l[i][k] * l[j][k];
			}
			
			comp_t app1;
			COMP_SUB(a[(offseti + i) * MATSIZE + offsetj + j], sum, app1);
			COMP_DIV(app1, l[j * BLOCKSIZE + j], l[i * BLOCKSIZE + j]);
			// l[i][j] = (a[i][j] - sum) / l[j][j];
		}
	}
}


/* Inversion of a block using Gauss algorithm without pivoting
 * (= LU factorization).
 * Returns the result in b.
 *
 * param c	pointer to source matrix (block matrix)
 * param bi	row index of the block to compute in matrix c
 * param bj	column index of the block to compute in matrix c
 * param b	pointer to destination block (ptr to block, not ptr to block matrix)
 */
void invert(const comp_t *c, unsigned bi, unsigned bj, comp_t *b)
{
	comp_t a[BLOCKSIZE][BLOCKSIZE];		// support matrix
	comp_t m, pivot;					// support variables
	int i, j, k;						// indexes used in loops
	unsigned offseti = bi * BLOCKSIZE;	// row offset for matrix c
	unsigned offsetj = bj * BLOCKSIZE;	// column offset for matrix c
	
	/* Initialization of matrices a and b (b is initialized to identity matrix) */
	for (i = 0; i < BLOCKSIZE; i++) {
		for (j = 0; j < BLOCKSIZE; j++) {
			a[i][j] = c[(offseti + i) * MATSIZE + offsetj + j];
			b[i * BLOCKSIZE + j].real = 0.0;
			b[i * BLOCKSIZE + j].imag = 0.0;
		}
		
		b[i * BLOCKSIZE + i].real = 1.0;
	}
	
	/* Ax = b  ->  LUx = b  ->  Ux = inv(L)b	*
     * First step: multiply by inv(L)			*/
	for (k = 0; k < BLOCKSIZE; k++) {
		pivot = a[k][k];
		//if (fabs(pivot) < 5e-16) return 0.0;
		
		for (i = k + 1; i < BLOCKSIZE; i++) {
			//m = a[i][k] / pivot;
			COMP_DIV(a[i][k], pivot, m);
			for (j = 0; j < BLOCKSIZE; j++) {
				//b[i][j] -= m * b[k][j];
				comp_t tmp;
				COMP_MUL(m, b[k * BLOCKSIZE + j], tmp);
				COMP_SUB(b[i * BLOCKSIZE + j], tmp, b[i * BLOCKSIZE + j]);
			}
			
			for (j = k; j < BLOCKSIZE; j++) {
				//a[i][j] -= m * a[k][j];
				comp_t tmp;
				COMP_MUL(m, a[k][j], tmp);
				COMP_SUB(a[i][j], tmp, a[i][j]);
			}
		}
	}
	
	/* Ux = inv(L)b  ->  x = inv(U)(inv(L)b)	*
     * Second step: multiply by inv(U)			*
     *              = backward substitution		*/
	for (i = BLOCKSIZE - 1; i >= 0; i--) {
		pivot = a[i][i];
		//if (fabs(pivot) < 5e-16) return 0.0;
		
		for (k = 0; k < BLOCKSIZE; k++) {
			m = b[i * BLOCKSIZE + k];
			for (j = i + 1; j < BLOCKSIZE; j++) {
				//m -= a[i][j] * b[j][k];
				comp_t tmp;
				COMP_MUL(a[i][j], b[j * BLOCKSIZE + k], tmp);
				COMP_SUB(m, tmp, m);
			}
			
			//b[i][k] = m / pivot;
			COMP_DIV(m, pivot, b[i * BLOCKSIZE + k]);
		}
	}
}


/* Multiplication between two blocks. Returns the result in res.
 *
 * param a	 pointer to source matrix (block matrix), first operand
 * param bi	 row index of the block to compute in matrix a
 * param bj	 column index of the block to compute in matrix a
 * param b	 pointer to block (ptr to block, not ptr to block matrix), second operand
 * param res pointer to destination block (ptr to block, not ptr to block matrix)
 */
void matrixMul(const comp_t *a, unsigned bi, unsigned bj,
               const comp_t *b, comp_t *res)
{
	unsigned i, j, k;					// indexes used in loops
	unsigned offseti = bi * BLOCKSIZE;	// row offset for matrix a
	unsigned offsetj = bj * BLOCKSIZE;	// column offset for matrix a
	
	for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
			res[i * BLOCKSIZE + j].real = res[i * BLOCKSIZE + j].imag = 0.0;
	
	for (i = 0; i < BLOCKSIZE; i++) {
		for (j = 0; j < BLOCKSIZE; j++) {
			for (k = 0; k < BLOCKSIZE; k++) {
				comp_t tmp;
				COMP_MUL(a[(offseti + i) * MATSIZE + offsetj + k], b[k * BLOCKSIZE + j], tmp);
				COMP_SUM(res[i * BLOCKSIZE + j], tmp, res[i * BLOCKSIZE + j]);
			}
		}
	}
	
}


/* Transposition of a block.
 *
 * param a	pointer to source matrix (block matrix)
 * param bi	row index of the block to compute in matrix a
 * param bj	column index of the block to compute in matrix a
 * param l	pointer to destination block (ptr to block, not ptr to block matrix)
 */
void transpose(const comp_t *a, unsigned bi, unsigned bj, comp_t *res)
{
	unsigned i, j;						// indexes used in loops
	unsigned offseti = bi * BLOCKSIZE;	// row offset for matrix a
	unsigned offsetj = bj * BLOCKSIZE;	// column offset for matrix a
	
	for (i = 0; i < BLOCKSIZE; i++) {
		for (j = 0; j < BLOCKSIZE; j++) {
			unsigned idx = j * BLOCKSIZE + i;
			
			res[idx] = a[(offseti + i) * MATSIZE + offsetj + j];
			res[idx].imag *= -1;
		}
		
	}
	
}

/* Subtraction between two blocks. Returns the result in res.
 *
 * param a	 pointer to source matrix (block matrix), first operand
 * param bia row index of the block to compute in matrix a
 * param bja column index of the block to compute in matrix a
 * param b	 pointer to block (ptr to block, not ptr to block matrix), second operand
 * param res pointer to destination matrix (block matrix)
 * param bir row index of the block to compute in matrix res
 * param bjr column index of the block to compute in matrix res
 */
void matrixSub(const comp_t *a, unsigned bia, unsigned bja,
               const comp_t *b, comp_t *res, unsigned bir, unsigned bjr)
{
	unsigned i, j;							// indexes used in loops
	unsigned offsetia = bia * BLOCKSIZE;	// row offset for matrix a
	unsigned offsetja = bja * BLOCKSIZE;	// column offset for matrix a
	unsigned offsetir = bir * BLOCKSIZE;	// row offset for matrix res
	unsigned offsetjr = bjr * BLOCKSIZE;	// column offset for matrix res
	
	for (i = 0; i < BLOCKSIZE; i++)
		for (j = 0; j < BLOCKSIZE; j++)
			COMP_SUB(a[(offsetia + i) * MATSIZE + offsetja + j],
			         b[i * BLOCKSIZE + j],
					 res[(offsetir + i) * MATSIZE + offsetjr + j]);
}



// generic worker
class Worker: public ff_node {
public:
	void *svc(void *task) {
        comp_t *l = ((ff_task_t *) task)->l;
		int i, k, j, cpi, cpj;		// indexes used in loops
		
		// Number of blocks
		int n = MATSIZE / BLOCKSIZE;
		
		/* Temporary block */
		comp_t tmp[BLOCKSIZE * BLOCKSIZE];
		
		// Begin factorization
		for (k = 0; k < n; k++) {
			// L[k][k] = factor(L[k][k])
			
			for (cpi = 0; cpi < BLOCKSIZE * BLOCKSIZE; cpi++)
				tmp[cpi].real = tmp[cpi].imag = 0.;
			
			// tmp = factor(l[k][k])
			factor(l, k, k, tmp);
			
			// l[k][k] = tmp
			for (cpi = 0; cpi < BLOCKSIZE; cpi++)
				for (cpj = 0; cpj < BLOCKSIZE; cpj++)
					l[(k * BLOCKSIZE + cpi) * MATSIZE + k * BLOCKSIZE + cpj] =
					tmp[cpi * BLOCKSIZE + cpj];
			
			// tmp = inv(l[k][k])
			invert(l, k, k, tmp);
			
			// tmp = transp(tmp)
			for (cpi = 0; cpi < BLOCKSIZE; cpi++) {
				for (cpj = cpi + 1; cpj < BLOCKSIZE; cpj++) {
					unsigned idx1 = cpi * BLOCKSIZE + cpj;
					unsigned idx2 = cpj * BLOCKSIZE + cpi;
					comp_t swap = tmp[idx1];
					tmp[idx1] = tmp[idx2];
					tmp[idx2] = swap;
					tmp[idx1].imag *= -1;
					tmp[idx2].imag *= -1;
				}
			}
			
			for (i = k + 1; i < n; i++) {
				comp_t lik[BLOCKSIZE * BLOCKSIZE];
				// lik = l[i][k] * tmp
				matrixMul(l, i, k, tmp, lik);
				
				// l[i][k] = lik
				for (cpi = 0; cpi < BLOCKSIZE; cpi++)
					for (cpj = 0; cpj < BLOCKSIZE; cpj++)
						l[(i * BLOCKSIZE + cpi) * MATSIZE + k * BLOCKSIZE + cpj] =
						lik[cpi * BLOCKSIZE + cpj];
			}
			
			for (j = k + 1; j < n; j++) {
				// L[j][j] =  L[j][j] – L[j][k] * (L[j][k])T
				
				comp_t ljkt[BLOCKSIZE * BLOCKSIZE];
				
				transpose(l, j, k, ljkt);
				matrixMul(l, j, k, ljkt, tmp);
				matrixSub(l, j, j, tmp, l, j, j);
				
				for (i = j + 1; i < n; i++) {
					// L[i][j] =  L[i][j] – L[i][k] * (L[j][k])T;
					
					matrixMul(l, i, k, ljkt, tmp);
					
					matrixSub(l, i, j, tmp, l, i, j);
				}	// End of i loop
				
			}	// End of j loop
			
		}	// End of k loop
		
		return GO_ON;
    }
	
};


// the load-balancer filter
class Emitter: public ff_node
{
private:
    int ntask;
	ff_task_t *tasks;
	
public:
    Emitter(ff_task_t *taskList, int max_task) : ntask(max_task) { tasks = taskList; }
	
    void *svc(void *) {
        if (--ntask < 0) return NULL;
        
		return tasks + ntask;
    }
};


int main(int argc, 
         char * argv[]) {
	
    if (argc < 4) {
        std::cerr << "usage: " 
		<< argv[0] 
		<< " nworkers streamlen inputfile\n";
        return -1;
    }
    
    int nworkers = atoi(argv[1]);
    int streamlen = atoi(argv[2]);
	char *infile = argv[3];
	
    if (nworkers <= 0 || streamlen <= 0) {
        std::cerr << "Wrong parameters values\n";
        return -1;
    }
	
	FILE *fp = fopen(infile, "r");
	if (!fp)
	{
		std::cerr << "Error opening input file\n";
		return -1;
	}
	
	// Array of tasks (each task is made up of an A matrix and a L matrix)
	ff_task_t *tasks = new ff_task_t[streamlen];
	
	// Variables needed to use HUGE_PAGES
	int huge_size;
	//char mem_file_name[32];
	//int fmem;
	char *mem_block;
	
	// HUGE_PAGES initialization
	huge_size = streamlen * MATSIZE * MATSIZE * sizeof(comp_t) * 2;
	/*huge_size = (huge_size + HUGE_PAGE_SIZE-1) & ~(HUGE_PAGE_SIZE-1);
	
	sprintf(mem_file_name, "/huge/huge_page_cbe.bin");
	assert((fmem = open(mem_file_name, O_CREAT | O_RDWR, 0755)) != -1);
	remove(mem_file_name);
	
	assert((mem_block = (char*) mmap(0, huge_size, PROT_READ | PROT_WRITE, MAP_PRIVATE, fmem, 0)) != MAP_FAILED);
	*/
	
	// Memory allocation without HUGE_PAGES
	mem_block = (char *) malloc(huge_size);
	
	tasks[0].a = (comp_t *) mem_block;
	tasks[0].l = (comp_t *) (mem_block + (MATSIZE * MATSIZE * sizeof(comp_t)));
	
	// Reads matrix A from input file and initializes matrix L
	int n;
	fscanf(fp, "%d", &n);
	assert(n == MATSIZE);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			fscanf(fp, "%f", &tasks[0].a[(i * MATSIZE) + j].real);
			fscanf(fp, "%f", &tasks[0].a[(i * MATSIZE) + j].imag);
			tasks[0].l[(i * MATSIZE) + j].real = tasks[0].a[(i * MATSIZE) + j].real;
			tasks[0].l[(i * MATSIZE) + j].imag = tasks[0].a[(i * MATSIZE) + j].imag;
		}
	
	fclose(fp);
	
	// Replication of matrices A and L to generate a stream of tasks
	for (int i = 1; i < streamlen; i++) {
		tasks[i].a = (comp_t *)
			(mem_block + ((i * 2) * MATSIZE * MATSIZE * sizeof(comp_t)));
		tasks[i].l = (comp_t *)
			(mem_block + ((i * 2 + 1) * MATSIZE * MATSIZE * sizeof(comp_t)));
		memcpy((comp_t *) tasks[i].a, (const comp_t *) tasks[0].a,
		       (size_t) MATSIZE * MATSIZE * sizeof(comp_t));
		memcpy((comp_t *) tasks[i].l, (const comp_t *) tasks[0].l,
		       (size_t) MATSIZE * MATSIZE * sizeof(comp_t));
	}
	
	// Farm declaration
    ff_farm<> farm;
    
	// Emitter declaration
    Emitter E(tasks, streamlen);
    farm.add_emitter(&E);
	
	// Workers declaration
    std::vector<ff_node *> w;
    for(int i = 0; i < nworkers; i++)
		w.push_back(new Worker);
    
	farm.add_workers(w);
	
	// Starts computation
	if (farm.run_and_wait_end() < 0) {
        error("running farm\n");
        return -1;
    }
    
    std::cout << "DONE, completion time = " << farm.ffTime() << " (ms),"
	<< " service time = " << (farm.ffTime() / streamlen) << " (ms)\n";
	
	
	// Prints the result matrix l[0] in a file
	const char *factorizationFileName = "./choleskyIntelBlock.txt";
	
	fp = fopen(factorizationFileName, "w+");
	if (fp == NULL) {
		perror("Error opening output file");
		return -1;
	}
	
	fprintf(fp, "Block algorithm farm version on Intel: L result matrix \n");
	
	for (int i = 0; i < MATSIZE; i++) {
		int j;
		fprintf(fp, "[ ");
		for (j = 0; j <= i ; j++) {
			fprintf(fp, "% 6.3f ", tasks[0].l[i * MATSIZE + j].real);
			fprintf(fp, "% 6.3fi ", tasks[0].l[i * MATSIZE + j].imag);
		}
		for ( ; j < MATSIZE; j++) {
			fprintf(fp, "% 6.3f ", 0.);
			fprintf(fp, "% 6.3fi ", 0.);
		}
		
		fprintf(fp, " ]\n");
	}
	
	fprintf(fp, "\n");
	
	fclose(fp);
	
	delete [] tasks;
	free(mem_block);	// without HUGE_PAGES
	
    return 0;
}
