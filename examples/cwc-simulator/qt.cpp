#include "qt.h"
#include "filter.h"
#include <list>
#include <limits>
#include <numeric>
#include <algorithm>
using namespace std;

double dist(point_t x, point_t y){
  double d_position = abs(x.position - y.position);
  double d_prediction = abs(x.prediction - y.prediction);
  return d_position / 3 + d_prediction * 2.0 / 3;  
}

point_t centroid(list<point_t > x){
  point_t res;
  res.position=0;
  res.prediction=0;
  int n=0;
  for(list<point_t>::iterator i = x.begin();i!=x.end();i++){
    res.position+=(*i).position;
    res.prediction+=(*i).prediction;
    n++;
  }
  res.position=res.position/n;
  res.prediction=res.prediction/n;
  return res;
}

double dist(list<point_t > x, list<point_t > y){
  double temp, d=.0;
  for(list<point_t >::iterator i=x.begin(); i!=x.end(); i++)
    for(list<point_t >::iterator j=y.begin(); j!=y.end(); j++)
      if((temp=dist(*i, *j))>d) d=temp;
  return d;
  //  return  dist(centroid(x),centroid(y));
}

pair<vector<point_t >, vector<list<int> > > qt(
					       vector<valarray<double> > & curves,
					       double threshold,
					       double time_win,
					       unsigned int window_size,
					       unsigned int degree,
					       double prediction_factor,
  					       unsigned int fw_left,
					       unsigned int fw_right
					       )
{
  unsigned int n_points = curves.size();
  point_t points;
  vector<list<point_t> > s(n_points);
  vector<list<int> > r(n_points);
  //compute points using the filter
  Savgol_Prediction savgol(fw_left,fw_right, degree, prediction_factor * time_win, time_win);
  for(unsigned int i=0; i<n_points; ++i) {
    points.position = savgol.filt(curves[i]);
    points.prediction = savgol.prediction(curves[i]);
    s[i].push_front(points);
    r[i].push_front(i);
  }


  //  cout << "fine caricamento" << endl << endl;

  vector<double> distanze((r.size()-1)*r.size()/2); //la matrice TRIANGOLARE ha n-1 righe e n-1 colonne
  vector<double>::iterator i_min;
  int ii, jj;

  //Calcolo iniziale delle distanze
  for(unsigned int i=0; i<distanze.size(); i++){
	
    ii=(int)(sqrt((double)(8*i+1))-1)/2;
    jj=i-ii*(ii+1)/2;
    distanze[i]=dist(*s[ii+1].begin(), *s[jj].begin());
    //  cout <<i << " "<< ii+1 <<" "<<" " << jj << " "<< distanze[i]<< endl;
    //   cout <<*s[ii+1].begin() << " " << *s[jj].begin() << " "<< distanze[i]<< endl;
  }

  //  cout.precision(2);

  while(*(i_min=min_element(distanze.begin(), distanze.end()))<threshold){
    //Selezione minimimo
    ii=(int)(sqrt((double)(8*(i_min-distanze.begin())+1))-1)/2;
    jj=(i_min-distanze.begin())-ii*(ii+1)/2;
 	
    //  cout << (ii+1) <<" " << jj << " " << *i_min <<endl;

    //Fusione liste ii entra in jj
    s[jj].splice(s[jj].begin(), s[ii+1]); s[ii+1].clear(); s.erase(s.begin()+ii+1);
    r[jj].splice(r[jj].begin(), r[ii+1]); r[ii+1].clear(); r.erase(r.begin()+ii+1);
	
    //    	for(int i=0, incr=1; i<distanze.size(); i+=incr, incr++){
    //	for(int j=i; j<i+incr; j++) cout << distanze[j]<< "\t";
    //	cout << endl;
    //	}
    	
    //Cancellare riga
    distanze.erase(distanze.begin()+ii*(ii+1)/2, distanze.begin()+(ii+1)*(ii+2)/2);

    //Cancellare colonna
    for(unsigned int i=(ii+1)*(ii+2)/2, incr=ii+1; i<distanze.size();i+=incr, incr++)
      distanze.erase(distanze.begin()+i); 
 
    //   	cout << "Cancellazione" << endl;
    //	for(int i=0, incr=1; i<distanze.size(); i+=incr, incr++){
    //	for(int j=i; j<i+incr; j++) cout << distanze[j]<< "\t";
    //	cout << endl;
    //	}
    
    //Ricalcolare riga
    ii=0;
    for(int i=jj*(jj-1)/2; i<jj*(jj+1)/2;i++)
      distanze[i]=dist(s[jj],s[ii++]); //TODO check: fatto 1° controllo OK
    if(distanze.size()<2) break;	
    //Ricalcolare colonna
    ii=jj+1;
    for(unsigned int i=jj*(jj+3)/2, incr=jj+1; i<distanze.size(); i+=incr, incr++)
      distanze[i]=dist(s[ii++],s[jj]);//TODO check: fatto 1° controllo OK

    //  	cout << "Ricalcolo" << endl;
    //	for(int i=0, incr=1; i<distanze.size(); i+=incr, incr++){
    //	for(int j=i; j<i+incr; j++) cout << distanze[j]<< "\t";
    //	cout << endl;
    //	}
    //exit(1);
  }

  //  cout<<"sto finendo"<<endl;
  //Carico centroidi cluster e Libero memoria
  vector<point_t> centers(s.size());
  for(unsigned int i=0; i<s.size(); i++){
    centers[i]=centroid(s[i]);
    s[i].clear();
  }
  s.empty();
  // cout<<"finito qt"<<endl;
  return make_pair(centers,r);
}


/*
  int main(int argc, char* argv[]){
  int i,j;
  int numCoords, numClusters;
  int numObjs;
  valarray<double>  prova; 
	
  cout << "Enter the number of features \n";
  cin >> numCoords;
  cout << "Enter the number of objects \n";
  cin >> numObjs;
  cout << "Enter the values by rows\n";
  prova.resize(numObjs*numCoords);
  for(i = 0; i < numObjs*numCoords; i++){
  cin >> prova[i];
  }
  cout << "Immissione corretta\n";
  pair<vector<valarray<double> >, vector<list<int> > > clusters=qt(prova,  numCoords, 2.0); 
  cout <<clusters.first.size()<<" clusters generated"<<endl;
  for(int i=0; i<clusters.first.size(); i++) {
  for(int j=0; j<numCoords; j++)
  cout << clusters.first[i][j]<<" ";
  cout<<endl;
  }
  return 0;
  }
*/


