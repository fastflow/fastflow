/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */

/**
  Fibonacci: computes the n-th number of the fibonacci sequence
  */

#include <iostream>
#include <functional>
#include <vector>
#include <ff/dc.hpp>
using namespace ff;



using namespace std;

/*
 * Operand and Result are just integers
 */

/*
 * Divide Function: recursively compute n-1 and n-2
 */
void divide(const unsigned int &op,std::vector<unsigned int> &subops)
{
	subops.push_back(op-1);
	subops.push_back(op-2);
}

/*
 * Base Case
 */
void seq(const unsigned int &op, unsigned int &res)
{
	res=1;
}

/*
 * Combine function
 */
void combine(vector<unsigned int>& res, unsigned int &ret)
{
	ret=res[0]+res[1];
}

/*
 * Condition for base case
 */
bool cond(const unsigned int &op)
{
	return (op<=2);
}



int main(int argc, char *argv[])
{

	if(argc<3)
	{
		fprintf(stderr,"Usage: %s <N> <pardegree>\n",argv[0]);
		exit(-1);
	}
	unsigned int start=atoi(argv[1]);
	int nwork=atoi(argv[2]);

	unsigned int res;
	//lambda version just for testing it
	ff_DC<unsigned int, unsigned int> dac(
				[](const unsigned int &op,std::vector<unsigned int> &subops){
						subops.push_back(op-1);
						subops.push_back(op-2);
					},
				[](vector<unsigned int>& res, unsigned int &ret){
						ret=res[0]+res[1];
					},
				[](const unsigned int &op, unsigned int &res){
						res=1;
					},
				[](const unsigned int &op){
						return (op<=2);
					},
				start,
				res,
				nwork
				);

	if (dac.run_and_wait_end()<0) {
        error("running dac\n");
        return -1;
    }
	printf("Result: %d\n",res);
}
