#include <iostream>
#include <fstream>
#include <cmath>
#include <cstdlib>

using namespace std;


int main(int argc, char **argv)
{
	char buf[256];
	char c;
	double a1;
	double b1;
	double a2;
	double b2;
	// counts the # of elements with an error bigger than threshold
	unsigned wrongCounter = 0;
	
	// program parameters check
	if (argc != 5) {
		cerr << "usage: " << argv[0] << " <dim> <file1> <file2> <threshold exp>" << endl << endl;
		exit(1);
	}
	
	int dim = atoi(argv[1]);
	ifstream file1(argv[2], ifstream::in);
	ifstream file2(argv[3], ifstream::in);
	double threshold = pow((double) 10, atoi(argv[4]));
	
	// drops the first line of each file
	file1.getline(buf, sizeof(buf));
	file2.getline(buf, sizeof(buf));
	
	for (int i = 0; i < dim; i++) {
		file1 >> c;	// [
		file2 >> c;	// [
		for (int j = 0; j < dim; j++) {
			// reads a complex number from both files in the form a + bi
			file1 >> a1;	// a
			file2 >> a2;	// a
			file1 >> b1;	// b
			file2 >> b2;	// b
			file1 >> c;	// i
			file2 >> c;	// i
		//	cout << "a1: " << a1 << "\tb1: " << b1 << "\ta2: " << a2 << "\tb2: " << b2
		//	     <<  "\tdiff real: " << fabs(a1 - a2) << "\tdiff imaginary: "
		//	     << fabs(b1 - b2) << endl;
			if (fabs(a1 - a2) > threshold || fabs(b1 - b2) > threshold) {
				cerr << "Difference found on row " << i << " and column " << j << endl;
				//cerr << "threshold: " << threshold << "\tdiff a: " << fabs(a1 - a2)
				//     << "\tdiff b: " << fabs(b1 - b2) << endl;
				cerr << "value in file1: " << a1 << " + " << b1 << "i" << endl;
				cerr << "value in file2: " << a2 << " + " << b2 << "i" << endl;
				cerr << "diff real: " << fabs(a1 - a2) << "\tdiff imaginary: "
				     << fabs(b1 - b2) << endl << endl;
				//exit(1);
				wrongCounter++;
			}

		}
		
		file1 >> c;	// ]
		file2 >> c;	// ]
	}

	file1.close();
	file2.close();
	
	cout << "Elements compared: " << dim * dim << "\t wrong values: " << wrongCounter << endl << endl;
	
	return 0;
}

