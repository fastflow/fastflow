#ifndef _CWC_FILTER_H_
#define _CWC_FILTER_H_
#include <valarray>
#include <iostream>
#include <cmath>
#include <string>
using namespace std;


typedef double DP;
typedef valarray<DP> Vec_DP, Vec_O_DP, Vec_IO_DP;
typedef const valarray<DP> Vec_I_DP;
typedef valarray<int> Vec_INT, Vec_O_INT, Vec_IO_INT;
typedef const valarray<int> Vec_I_INT;

//SavgolMat
template <class T>
class SavgolMat {
private:
  int nn;
  int mm;
  T **v;
public:
  SavgolMat();
  SavgolMat(int n, int m);			// Zero-based array
  SavgolMat(const T &a, int n, int m);	//Initialize to constant
  SavgolMat(const T *a, int n, int m);	// Initialize to array
  SavgolMat(const SavgolMat &rhs);		// Copy constructor
  SavgolMat & operator=(const SavgolMat &rhs);	//assignment
  SavgolMat & operator=(const T &a);		//assign a to every element
  inline T* operator[](const int i);	//subscripting: pointer to row i
  inline const T* operator[](const int i) const;
  inline int nrows() const;
  inline int ncols() const;
  ~SavgolMat();
};

template <class T>
SavgolMat<T>::SavgolMat() : nn(0), mm(0), v(0) {}

template <class T>
SavgolMat<T>::SavgolMat(int n, int m) : nn(n), mm(m), v(new T*[n])
{
  v[0] = new T[m*n];
  for (int i=1; i< n; i++)
    v[i] = v[i-1] + m;
}

template <class T>
SavgolMat<T>::SavgolMat(const T &a, int n, int m) : nn(n), mm(m), v(new T*[n])
{
  int i,j;
  v[0] = new T[m*n];
  for (i=1; i< n; i++)
    v[i] = v[i-1] + m;
  for (i=0; i< n; i++)
    for (j=0; j<m; j++)
      v[i][j] = a;
}

template <class T>
SavgolMat<T>::SavgolMat(const T *a, int n, int m) : nn(n), mm(m), v(new T*[n])
{
  int i,j;
  v[0] = new T[m*n];
  for (i=1; i< n; i++)
    v[i] = v[i-1] + m;
  for (i=0; i< n; i++)
    for (j=0; j<m; j++)
      v[i][j] = *a++;
}

template <class T>
SavgolMat<T>::SavgolMat(const SavgolMat &rhs) : nn(rhs.nn), mm(rhs.mm), v(new T*[nn])
{
  int i,j;
  v[0] = new T[mm*nn];
  for (i=1; i< nn; i++)
    v[i] = v[i-1] + mm;
  for (i=0; i< nn; i++)
    for (j=0; j<mm; j++)
      v[i][j] = rhs[i][j];
}

template <class T>
SavgolMat<T> & SavgolMat<T>::operator=(const SavgolMat<T> &rhs)
// postcondition: normal assignment via copying has been performed;
//		if matrix and rhs were different sizes, matrix
//		has been resized to match the size of rhs
{
  if (this != &rhs) {
    int i,j;
    if (nn != rhs.nn || mm != rhs.mm) {
      if (v != 0) {
	delete[] (v[0]);
	delete[] (v);
      }
      nn=rhs.nn;
      mm=rhs.mm;
      v = new T*[nn];
      v[0] = new T[mm*nn];
    }
    for (i=1; i< nn; i++)
      v[i] = v[i-1] + mm;
    for (i=0; i< nn; i++)
      for (j=0; j<mm; j++)
	v[i][j] = rhs[i][j];
  }
  return *this;
}

template <class T>
SavgolMat<T> & SavgolMat<T>::operator=(const T &a)	//assign a to every element
{
  for (int i=0; i< nn; i++)
    for (int j=0; j<mm; j++)
      v[i][j] = a;
  return *this;
}

template <class T>
inline T* SavgolMat<T>::operator[](const int i)	//subscripting: pointer to row i
{
  return v[i];
}

template <class T>
inline const T* SavgolMat<T>::operator[](const int i) const
{
  return v[i];
}

template <class T>
inline int SavgolMat<T>::nrows() const
{
  return nn;
}

template <class T>
inline int SavgolMat<T>::ncols() const
{
  return mm;
}

template <class T>
SavgolMat<T>::~SavgolMat()
{
  if (v != 0) {
    delete[] (v[0]);
    delete[] (v);
  }
}

typedef SavgolMat<DP> Mat_DP, Mat_O_DP, Mat_IO_DP;
typedef const SavgolMat<DP> Mat_I_DP;




//run-time error
namespace Savgol {
  inline void filter_error(const string error_text)
  //error handler
  {
    cerr << "Savitzky-Golay filter run-time error..." << endl;
    cerr << error_text << endl;
    cerr << "...now exiting to system..." << endl;
    exit(1);
  }
}





//filter function
namespace Savgol {
  void ludcmp(Mat_IO_DP &a, Vec_O_INT &indx, DP &d);
  void lubksb(Mat_I_DP &a, Vec_I_INT &indx, Vec_IO_DP &b);
  void savgol(Vec_O_DP &c, const int np, const int nl, const int nr,
	      const int ld, const int m);
}



//filter wrapper
class Savgol_Prediction{
	
public:
  Savgol_Prediction(const int nl, const int nr, const int m, DP time_step, DP time_win );
  double prediction(Vec_I_DP& x);
  double filt(Vec_I_DP& x);
  bool   peak(Vec_I_DP&, DP, DP);

private:
  Vec_O_DP c,cp,c1,c2;
};
#endif
