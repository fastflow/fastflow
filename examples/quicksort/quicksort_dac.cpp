/**
  Quicksort: sort an array of N integer using the Divide&Conquer (DAC) pattern

  Author: Tiziano De Matteis <dematteis@di.unipi.it>
  */

#include <iostream>
#include <functional>
#include <algorithm>
#include <ff/dc.hpp>
using namespace ff;
using namespace std;
#define CUTOFF 2000


struct ops {
    int *array=nullptr;          //array to sort
    int left=0;                  //left index
    int right=0;                 //right index

};

typedef struct ops Problem;
typedef struct ops Result;

/* ------------------------------------------------------------------ */
//maximum value for arrays elements
const int MAX_NUM=99999.9;

static bool isArraySorted(int *a, int n) {
    for(int i=1;i<n;i++)
        if(a[i]<a[i-1])
            return false;
    return true;
}
static int *generateRandomArray(int n) {
    srand ((time(0)));
    int *numbers=new int[n];
    for(int i=0;i<n;i++)
        numbers[i]=(int) (rand()) / ((RAND_MAX/MAX_NUM));
    return numbers;
}
/* ------------------------------------------------------------------ */


/*
 * The divide chooses as pivot the middle element and redistributes the elements
 */
void divide(const Problem &op, std::vector<Problem> &ops)
{
    ops.push_back(Problem());
    ops.push_back(Problem());

    int *a=op.array;
    int pivot=a[(op.left+op.right)/2];
    int i = op.left-1, j = op.right+1;
    int tmp;

    while(true)
    {
        do{
            i++;
        }while(a[i]<pivot);
        do{
            j--;
        }while(a[j]>pivot);

        if(i>=j)
           break;

        //swap
        tmp=a[i];
        a[i]=a[j];
        a[j]=tmp;
    }

    //j is the pivot

    ops[0].array=a;
    ops[0].left=op.left;
    ops[0].right=j;

    ops[1].array=a;
    ops[1].left=j+1;
    ops[1].right=op.right;
}


/*
 * The Combine does nothing
 */
void mergeQS(vector<Result> &ress, Result &ret)
{
    ret.array=ress[0].array;
    ret.left=ress[0].left;
    ret.right=ress[1].right;
}


/*
 * Base case: we resort on std::sort
 */
void seq(const Problem &op, Result &ret)
{

    std::sort(&(op.array[op.left]),&(op.array[op.right+1]));

	//build result
    ret.array=op.array;
    ret.left=op.left;
    ret.right=op.right;
}

/*
 * Base case condition
 */
bool cond(const Problem &op)
{
	return (op.right-op.left<=CUTOFF);
}

int main(int argc, char *argv[])
{
	if(argc<2)
	{
		cerr << "Usage: "<<argv[0]<< " <num_elements> <num_workers>"<<endl;
		exit(-1);
	}
	std::function<void(const Problem &,vector<Problem> &)> div(divide);
	std::function <void(const Problem &,Result &)> sq(seq);
	std::function <void(vector<Result >&,Result &)> mergef(mergeQS);
	std::function<bool(const Problem &)> cf(cond);

	int num_elem=atoi(argv[1]);
	int nwork=atoi(argv[2]);
	int *numbers=generateRandomArray(num_elem);

	Problem op;
	op.array=numbers;
	op.left=0;
	op.right=num_elem-1;
	Result res;
	ff_DC<Problem, Result> dac(div,mergef,sq,cf,op,res,nwork);
	unsigned long start_t = getusec();
	dac.run_and_wait_end();
	unsigned long end_t = getusec();
	if(!isArraySorted(numbers,num_elem)) {
	    fprintf(stderr,"Error: array is not sorted!!\n");
	    exit(-1);
	}
	printf("Time (usecs): %ld\n",end_t-start_t);

	return 0;
}
