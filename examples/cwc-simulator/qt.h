#ifndef QT_H
#define QT_H
#include <list>
#include <vector>
#include <valarray>
#include <iostream>
#include "definitions.h"
using namespace std;

typedef vector<point_t>  qt_center_t;
typedef pair<qt_center_t, vector<list<int> > > qt_result_t;

qt_result_t qt(vector<valarray<double> >& curves, double threshold, double time_win, unsigned int window_size, unsigned int, double,unsigned int, unsigned int);
ostream &operator<<(ostream &os, qt_center_t cs);
#endif


