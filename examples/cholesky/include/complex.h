#ifndef __COMPLEX_H__
#define __COMPLEX_H__

/* Struct to represent a complex number */
typedef struct {
	float real;
	float imag;
} comp_t;

// DEBUG
int operationCounter = 0;

// Add operator for two complex numbers
#define COMP_SUM(comp1, comp2, res) { \
(res).real = (comp1).real + (comp2).real; \
(res).imag = (comp1).imag + (comp2).imag; \
operationCounter++;	\
}

// Subtract operator for complex numbers
#define COMP_SUB(comp1, comp2, res) { \
(res).real = (comp1).real - (comp2).real; \
(res).imag = (comp1).imag - (comp2).imag; \
operationCounter++;	\
}

// Multiply operator for complex numbers
#define COMP_MUL(comp1,comp2,res) { \
(res).real = ((comp1).real * (comp2).real) - ((comp1).imag * (comp2).imag); \
(res).imag = ((comp1).imag * (comp2).real) + ((comp1).real * (comp2).imag); \
operationCounter++;	\
}

// Divide operator for complex numbers: [A*C + B*D + i(C*B – A*D)] / C2 + D2                          
#define COMP_DIV(comp1,comp2,res) { \
(res).real = (((comp1).real * (comp2).real) + ((comp1).imag * (comp2).imag)) / (((comp2).real * (comp2).real) + ((comp2).imag * (comp2).imag)); \
(res).imag = (((comp2).real * (comp1).imag) - ((comp1).real * (comp2).imag)) / (((comp2).real * (comp2).real) + ((comp2).imag * (comp2).imag)); \
operationCounter++; \
}

#endif	/* __COMPLEX_H__ */
