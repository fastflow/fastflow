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
 * Cholesky matrix factorization algorithm, standard version.
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

// generic worker
class Worker: public ff_node {
public:
    void *svc(void *task) {
		comp_t *a = ((task_t *) task)->a;
		comp_t *l = ((task_t *) task)->l;
		int i, k, j;	// indexes used in loops
		float sumSQR;	// support variable
		comp_t sum;		// support variable
		int idx;		// index of the current element
		
		for (j = 0; j < MATSIZE; j++) {
			sumSQR = 0.;
			
			for (k = 0; k < j; k++) {
				idx = j * MATSIZE + k;	// l[j][k] index
				//sum += l[j][k] * l[j][k];
				sumSQR += ((l[idx].real) * (l[idx].real) +
				           (l[idx].imag) * (l[idx].imag));
			}
			
			idx = j * MATSIZE + j;	// a[j][j] and l[j][j] index
			//sum = a[j][j] - sum;	// L[j][j]^2
			sumSQR = a[idx].real - sumSQR;
			//l[j][j] = sqrt(sum);
			l[idx].real = sqrt(sumSQR);
			
			for (i = j + 1; i < MATSIZE; i++) {
				sum.real = 0.;
				sum.imag = 0.;
				
				for (k = 0; k < j; k++) {
					//sum += l[i][k] * l[j][k];
					comp_t tmp;
					comp_t trasp;
					idx = j * MATSIZE + k;	// l[j][k] index
					trasp.real = l[idx].real;
					trasp.imag = -(l[idx].imag);
					COMP_MUL(l[i * MATSIZE + k], trasp, tmp);
					COMP_SUM(sum, tmp, sum);
				}
				
				//l[i][j] = (a[i][j] - sum) / l[j][j];
				comp_t app1;
				idx = i * MATSIZE + j;	// a[i][j] and l[i][j] index
				COMP_SUB(a[idx], sum, app1);
				COMP_DIV(app1, l[j * MATSIZE + j], l[idx]);
			}
		}
		
		return GO_ON;
    }
	
};


// the load-balancer filter
class Emitter: public ff_node
{
private:
    int ntask;
	task_t *tasks;

public:
    Emitter(task_t *taskList, int max_task) : ntask(max_task) { tasks = taskList; }

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
	task_t *tasks = new task_t[streamlen];
	
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
	
	// Reads matrix A from input file
	int n;
	fscanf(fp, "%d", &n);
	assert(n == MATSIZE);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < n; j++) {
			fscanf(fp, "%f", &tasks[0].a[(i * MATSIZE) + j].real);
			fscanf(fp, "%f", &tasks[0].a[(i * MATSIZE) + j].imag);
		}
	
	fclose(fp);
	
	// Initialization of matrix L
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			tasks[0].l[(i * MATSIZE) + j].real = 0.;
			tasks[0].l[(i * MATSIZE) + j].imag = 0.;
		}
	}
	
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
    Emitter e(tasks, streamlen);
    farm.add_emitter(&e);
	
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
	const char *factorizationFileName = "./choleskyIntelStandard.txt";
	
	fp = fopen(factorizationFileName, "w+");
	if (fp == NULL) {
		perror("Error opening output file");
		return -1;
	}
	
	fprintf(fp, "Standard algorithm farm version on Intel: L result matrix \n");
	
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
