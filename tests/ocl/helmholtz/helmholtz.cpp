/*
 * helmholtz.cpp
 *
 *  Created on: Dec 10, 2015
 *      Author: mdrocco
 */

#include "../ctest.h"

#include <ff/stencilReduceOCL.hpp>
#include <ff/utils.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>

struct const_t {
	const_t(int n, int m, float ax, float ay, float b, float omega) :
			n(n), m(m), ax(ax), ay(ay), b(b), omega(omega) {
	}

	int n, m;
	float ax, ay, b, omega;
};

#define ITERS 10

/******************************************************
 * Initializes data
 * Assumes exact solution is u(x,y) = (1-x^2)*(1-y^2)
 *
 ******************************************************/
void initialize(int n, int m, float alpha, float *dx, float *dy, float *u, float *f) {
	int xx, yy;

	*dx = 2.0 / (n - 1);
	*dy = 2.0 / (m - 1);

	/* Initilize initial condition and RHS */
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			xx = -1.0 + *dx * (i - 1);
			yy = -1.0 + *dy * (j - 1);
			u[j * n + i] = 0.0;
			f[j * n + i] = -alpha * (1.0 - xx * xx) * (1.0 - yy * yy) - 2.0 * (1.0 - xx * xx) - 2.0 * (1.0 - yy * yy);
		}
	}

}

/************************************************************
 * Checks error between numerical and exact solution
 *
 ************************************************************/
void error_check(int n, int m, float alpha, float dx, float dy, float *u, float *f) {
	float xx, yy, temp, error;
	dx = 2.0 / (n - 1);
	dy = 2.0 / (n - 2);
	error = 0.0;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			xx = -1.0 + dx * (i - 1);
			yy = -1.0 + dy * (j - 1);
			temp = u[j * n + i] - (1.0 - xx * xx) * (1.0 - yy * yy);
			error += temp * temp;
		}
	}
	error = sqrt(error) / (n * m);
	printf("Solution Error : %g\n", error);
}

class HelmholtzTask: public ff::baseOCLTask_2D<HelmholtzTask, float> {
public:
	HelmholtzTask(int n, int m, float dx, float dy, float alpha, float omega, float *u, float *f, int iters_) :
			w(n), h(m), u(u), f(f), Error(0.0), iters(iters_) {

		const float ax = 1.0 / (dx * dx); /* X-direction coef */
		const float ay = 1.0 / (dy * dy); /* Y_direction coef */
		const float b = -2.0 / (dx * dx) - 2.0 / (dy * dy) - alpha; /* Central coeff */

		std::cout << "ax = " << ax << ", ay = " << ay << ", b = " << b << ", omega = " << omega << std::endl;

		Const = new const_t(n, m, ax, ay, b, omega);
		uold = (float*) malloc(m * n * sizeof(float));
		memcpy(uold, u, m * n * sizeof(float));
	}

	HelmholtzTask() :
			h(-1), w(-1), u(nullptr), uold(nullptr), Const(nullptr), f(nullptr), Error(0.0), iters(0) {
	}

	~HelmholtzTask() {
		if (Const)
			delete Const;
		if (uold)
			free(uold);
	}

	void setTask(HelmholtzTask* t) {
		assert(t);
		const HelmholtzTask &task = *t;

		ff::MemoryFlags mfout(ff::CopyFlags::COPY, ff::ReuseFlags::DONTREUSE, ff::ReleaseFlags::DONTRELEASE);

		w = task.w;
		h = task.h;
		iters = task.iters;

		setHeight(h);
		setWidth(w);
		setInPtr(task.uold, w * h);
		setOutPtr(task.u, w * h);
		setReduceVar(&(task.Error));

		setEnvPtr(task.Const, 1);
		setEnvPtr(task.f, h * w);
	}

	bool iterCondition(const float &E, size_t iter) {
//		Error = sqrt(E) / (n * m);
//		return (Error > tol);
		return iter < iters;
	}

	int iters;

private:
	int w, h;
	float *u;
	float *f;
	float Error;
	const_t *Const;
	float *uold;
};

FF_OCL_REDUCE_COMBINATOR(reducef, float, x, y, return (x+y));

/*
 * API:
 * 'name' is the name of the string variable in which the code is stored
 * 'inT' is the element type of the input
 * 'height' is the number of rows in the input array
 * 'width' is the number of columns in the input array
 * 'row' is the row-index
 * 'col' is the column-index
 * 'env1T' is the element type of the first constant environment
 * 'env2T' is the element type of the second constant environment
 * 'code' is the OpenCL code of the elemental function
 */
FF_OCL_STENCIL_ELEMFUNC_2D_2ENV(stencilf, float, h, w, r, c, const_t, float,
	if(r > 0 && c > 0 && r < (h-1) && c < (w-1)) {
		const float ax = GET_ENV1(0,0).ax;
		const float ay = GET_ENV1(0,0).ay;
		const float b = GET_ENV1(0,0).b;
		const float omega = GET_ENV1(0,0).omega;

		float resid = ax * (GET_IN(r,c-1) + GET_IN(r,c+1));
		resid += ay * (GET_IN(r+1,c) + GET_IN(r-1,c));
		resid += b * GET_IN(r,c) - GET_ENV2(r,c);
		resid /= b;
		return GET_IN(r,c) - omega * resid;
	}

	return 0;
);

int main(int argc, char **argv) {
	float *u, *f, dx, dy;
	float dt, mflops;
	double st;
	int n, m, mits, iters;
	float tol, relax, alpha;

	printf("use: %s n m alpha relax tot mits iters\n", argv[0]);
	printf(" example %s 5000 5000 0.8 1.0 1e-7 1000 10\n", argv[0]);

	n = 5000;
	m = 5000;
	alpha = 0.8;
	relax = 1.0;
	tol = 1e-7;
	mits = 1000;
	iters = 10;

	if(argc>1) n = atoi(argv[1]);
	if(argc>2) m = atoi(argv[2]);
	if(argc>3) alpha = atof(argv[3]);
	if(argc>4) relax = atof(argv[4]);
	if(argc>5) tol = atof(argv[5]);
	if(argc>6) mits = atoi(argv[6]);
	if(argc>7) iters = atoi(argv[7]);

	//allocate
	u = (float *) malloc(n * m * sizeof(float));
	assert(u);
	f = (float *) malloc(n * m * sizeof(float));
	assert(f);

	//init
	initialize(n, m, alpha, &dx, &dy, u, f);
	float *utmp = (float *) malloc(n * m * sizeof(float));
	assert(utmp);
	memcpy(utmp, u, n * m * sizeof(float));

	//prepare task
	HelmholtzTask jt(n, m, dx, dy, alpha, relax, u, f, iters);

	//one-shot srl
	ff::ff_stencilReduceLoopOCL_2D<HelmholtzTask> helmholtz(jt, stencilf, reducef, 0.0);
	SET_DEVICE_TYPE(helmholtz);

	/* Solve Helmholtz equation */
	printf("Jacobi started\n");
	ff::ffTime(ff::START_TIME);
	helmholtz.run_and_wait_end();
	ff::ffTime(ff::STOP_TIME);
	dt = ff::ffTime(ff::GET_TIME);
	st = ff::diffmsec(helmholtz.getwstoptime(),helmholtz.getwstartime());

	printf("Total Number of Iterations %d\n", (int)jt.iters);

	printf("elapsed time : %12.6f  (ms)\n", dt);
	printf("svc time : %12.6g  (ms)\n", st);

	mflops = (0.000001 * mits * (m - 2) * (n - 2) * 13) / (dt / 1000.0);
	printf(" MFlops       : %12.6g (%d, %d, %d, %g)\n", mflops, mits, m, n, (dt / 1000.0));

	error_check(n, m, alpha, dx, dy, u, f);

	//diff
	float diff = 0;
	int ndiff = 0;
	for (int j = 0; j < m; j++) {
		for (int i = 0; i < n; i++) {
			if (u[j * n + i] != utmp[j * n + i]) {
				++ndiff;
				diff += std::abs(u[j * n + i] - utmp[j * n + i]);
			}
		}
	}
	std::cout << "total diff = " << diff << " over " << ndiff << " points\n";

	//clean-up
	free(u);
	free(f);
	free(utmp);

	return 0;
}

