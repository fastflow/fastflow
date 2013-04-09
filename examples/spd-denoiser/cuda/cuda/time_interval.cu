// Class for some time-related stuff...

#include "time_interval.h"
#include <ctime>
#include <iostream>

using namespace std;

// Set the initial time for the stopwatch
void TimeInterval::tic()
{
	tic_time = clock();
}

// Set the final time for the stopwatch and display the time interval
double TimeInterval::toc()
{
	long stop_time = clock();
	long diff = stop_time - tic_time;
	return ((double) diff)/CLOCKS_PER_SEC; // seconds
}
