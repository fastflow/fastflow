// Class for some time-related stuff...

#ifndef TIME_INTERVAL_H
#define TIME_INTERVAL_H

#include <string>

using namespace std;

class TimeInterval
{
	long tic_time;

	public:

	// Set the initial time for the stopwatch
	void tic();

	// Set the final time for the stopwatch and return the elapsed time interval
	double toc();
};

#endif
