/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */
/*   File:         seq_kmeans.c  (sequential version)                        */
/*   Description:  Implementation of simple k-means clustering algorithm     */
/*                 This program takes an array of N data objects, each with  */
/*                 M coordinates and performs a k-means clustering given a   */
/*                 user-provided value of the number of clusters (K). The    */
/*                 clustering results are saved in 2 arrays:                 */
/*                 1. a returned array of size [K][N] indicating the center  */
/*                    coordinates of K clusters                              */
/*                 2. membership[N] stores the cluster center ids, each      */
/*                    corresponding to the cluster a data object is assigned */
/*                                                                           */
/*   Author:  Wei-keng Liao                                                  */
/*            ECE Department, Northwestern University                        */
/*            email: wkliao@ece.northwestern.edu                             */
/*   Copyright, 2005, Wei-keng Liao                                          */
/*                                                                           */
/* * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * */

#include <stdio.h>
#include <stdlib.h>
#include "kmeans.h"

#include <iostream>
using namespace std;


/*----< euclid_dist_2() >----------------------------------------------------*/
/* square of Euclid distance between two multi-dimensional points            */
//__inline static
double euclid_dist_2(int    numdims,  /* no. dimensions */
                    double *coord1,   /* [numdims] */
                    double *coord2)   /* [numdims] */
{
  int i;
  double ans=0.0;

  for (i=0; i<numdims; i++)
    ans += (coord1[i]-coord2[i]) * (coord1[i]-coord2[i]);

  return(ans);
}

/*----< find_nearest_cluster() >---------------------------------------------*/
//__inline static
int find_nearest_cluster(int     numClusters, /* no. clusters */
                         int     numCoords,   /* no. coordinates */
                         double  *object,      /* [numCoords] */
                         double **clusters)    /* [numClusters][numCoords] */
{
  int   index, i;
  double dist, min_dist;

  /* find the cluster id that has min distance to object */
  index    = 0;
  min_dist = euclid_dist_2(numCoords, object, clusters[0]);

  for (i=1; i<numClusters; i++) {
    dist = euclid_dist_2(numCoords, object, clusters[i]);
    /* no need square root */
    if (dist < min_dist) { /* find the min and its array index */
      min_dist = dist;
      index    = i;
    }
  }
  return(index);
}

/*----< seq_kmeans() >-------------------------------------------------------*/
/* return an array of cluster centers of size [numClusters][numCoords]       */
double** seq_kmeans(double **objects,      /* in: [numObjs][numCoords] */
                   int     numCoords,    /* no. features for each object */
                   int     numObjs,      /* no. objects */
                   int     numClusters,  /* no. clusters */
                   double   threshold,    /* % objects change membership */
                   int    *membership,   /* out: [numObjs] */
		   int *sizes, //sizes of the clusters
		   double  *guessed_centers) //guessed initial centers (for the 1st feature)
{
  int      i, j, index, loop=0;
  int     *newClusterSize; /* [numClusters]: no. objects assigned in each
			      new cluster */
  double    delta;          /* % of objects change their clusters */
  double  **clusters;       /* out: [numClusters][numCoords] */
  double  **newClusters;    /* [numClusters][numCoords] */

  /* allocate a 2D space for returning variable clusters[] (coordinates
     of cluster centers) */
  clusters    = (double**) malloc(numClusters * sizeof(double*));
  assert(clusters != NULL);
  clusters[0] = (double*)  malloc(numClusters * numCoords * sizeof(double));
  assert(clusters[0] != NULL);
  for (i=1; i<numClusters; i++)
    clusters[i] = clusters[i-1] + numCoords;

  /* pick first numClusters elements of objects[] as initial cluster centers*/
  /*
    for (i=0; i<numClusters; i++)
    for (j=0; j<numCoords; j++)
    clusters[i][j] = objects[i][j];
  */

  /* use guessed centers as initial cluster centers*/
  for (i=0; i<numClusters; i++)
    clusters[i][0] = guessed_centers[i];

  /* initialize membership[] */
  for (i=0; i<numObjs; i++) membership[i] = -1;

  /* need to initialize newClusterSize and newClusters[0] to all 0 */
  newClusterSize = (int*) calloc(numClusters, sizeof(int));
  assert(newClusterSize != NULL);

  newClusters    = (double**) malloc(numClusters *            sizeof(double*));
  assert(newClusters != NULL);
  newClusters[0] = (double*)  calloc(numClusters * numCoords, sizeof(double));
  assert(newClusters[0] != NULL);
  for (i=1; i<numClusters; i++)
    newClusters[i] = newClusters[i-1] + numCoords;

  do {
    delta = 0.0;
    for (i=0; i<numObjs; i++) {
      /* find the array index of nestest cluster center */
      index = find_nearest_cluster(numClusters, numCoords, objects[i],
				   clusters);

      /* if membership changes, increase delta by 1 */
      if (membership[i] != index) delta += 1.0;

      /* assign the membership to object i */
      membership[i] = index;

      /* update new cluster centers : sum of objects located within */
      newClusterSize[index]++;
      for (j=0; j<numCoords; j++)
	newClusters[index][j] += objects[i][j];
    }

    //for (i=0; i<numClusters; i++) {
    //printf(" Cluster size %d\n",newClusterSize[i]);
    //}

    /* average the sum and replace old cluster centers with newClusters */
    for (i=0; i<numClusters; i++) {
      for (j=0; j<numCoords; j++) {
	if (newClusterSize[i] > 0)
	  clusters[i][j] = newClusters[i][j] / newClusterSize[i];
	newClusters[i][j] = 0.0;   /* set back to 0 */
      }
      sizes[i] = newClusterSize[i];
      newClusterSize[i] = 0;   /* set back to 0 */
    }
		
    //printf("turn %d delta1 %f delta2 %f\n",loop,delta,delta/numObjs);
    delta /= numObjs;
  } while (delta > threshold && loop++ < 500);

  free(newClusters[0]);
  free(newClusters);
  free(newClusterSize);

  return clusters;
}

