// Salt-and-Pepper_Noise_Removal_GRAYSCALE
// Filtro a due passi a più cicli per la rimozione del disturbo 'Salt & Pepper' da immagini bitmap a 8 bit in scala di grigi.
#if !(defined(FLAT) || defined(BORDER) || defined(CLUSTER) || defined(CUDA))
#define CONTINOUS 1
#endif

//#if (defined(CUDA) || defined(CLUSTER))
#ifdef CLUSTER
#define SEQ_DETECTION 1
#endif

#ifdef CUDA
#define NO_BLOCKS 1
#endif

#ifndef CUDA
#define FF_ACCEL 1
#endif

#include <stdlib.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <algorithm>
#include "definitions.h"
#include "bitmap.hpp"
#include "control_structures.hpp"
#include "noise_detection.hpp"

#ifdef CLUSTER
#include "clusters.hpp"
#else
#include "block.hpp"
#endif

#include "convergence.hpp"
#include "ff_accel.hpp"
#include "bmp_finalizer.hpp"
#include "parameters.h"
#include <iostream>
#include <vector>
#include <map>
#include <limits>
//#include <ff/platforms/platform.h>
#include "utils.h"
#include <ff/farm.hpp>
#include <ff/node.hpp>
#ifdef CUDA
#include <cuda_denoiser.hpp>
#endif
using namespace std;

#define EPS_STOP 1e-04 //convergence epsilon
#define K_BLOCKS 4


int main(int argc,char* argv[]) {
  
  //cerr << "Size of " << sizeof(struct bmp_point<char>) << "\n"; 
  //check flags consinstency
#if defined(CLUSTER) and defined(BORDER)
  cerr << "CLUSTER-BORDER conflict" << endl;
  exit(1);
#elif defined(BORDER) and defined(FLAT)
  cerr << "BORDER-FLAT conflict" << endl;
  exit(1);
#elif defined(CUDA) && (defined(BORDER) || defined(FLAT) || defined(CLUSTER) || defined(ALTERNATE))
  cerr << "CUDA-* conflict" << endl;
  exit(1);
#endif

#ifdef TIME
  long int t_finding, t_correction;
  long int t_whole = get_usec_from(0);
#ifndef NO_BLOCKS
  long int t_clustering;
#endif
  //long int t_red = 0;
#endif

  //parse command-line arguments
  string fname, out_fname;
  float alfa, beta;
  unsigned int w_max, max_cycles;
  bool fixed_cycles;
  unsigned int nworkers/*, set_fraction*/;
  bool verbose;
  arguments args;
  bool user_out_fname;
  get_arguments(argv, argc, args);
  fname = args.fname;
  alfa = args.alfa;
  beta = args.beta;
  w_max = args.w_max;
  max_cycles = args.max_cycles;
  fixed_cycles = args.fixed_cycles;
  nworkers = args.n_workers;
  user_out_fname = args.user_out_fname;
  if(user_out_fname)
    out_fname = args.out_fname;
  verbose = args.verbose;

  //open the bmp file
  FILE *fp;
  if(!(fp = fopen(fname.c_str(), "rb"))) {
    cerr << "ERROR: could not open the file " << fname << endl;
    exit(1);
  }

  //build output fname
  string prefix;
  if(user_out_fname)
    prefix = out_fname.substr(0, out_fname.length() - 4);
  else {
    prefix.append("RESTORED_");
    string trunked_fname = get_fname(fname);
    prefix.append(trunked_fname.substr(0, trunked_fname.length() - 4));
#if not defined(CLUSTER)
    prefix.append("_block");
#else
    prefix.append("_cluster");
#endif
#if defined(BORDER)
    prefix.append("_border");
#endif
#if defined(FLAT)
    prefix.append("_flat");
#endif
#if defined(ALTERNATE)
    prefix.append("_alternate");
#endif
  }

  //verbose headers
  if(verbose) {
    cout << "*** This is Salt & Pepper denoiser" << endl
	 << "mode: "
#ifndef CUDA
	 << "FF "
#if not defined(CLUSTER)
	 << "block "
#if not defined(BORDER) and not defined(FLAT)
	 << "non-deterministic "
#endif
#else
	 << "cluster "
#endif
#if defined(BORDER)
	 << "border "
#endif
#if defined(FLAT)
	 << "flat "
#endif
#if defined(ALTERNATE)
	 << "alternate "
#endif
#else //CUDA
	 << "CUDA "
#ifdef CUDA_PINNED_MEMORY
	 << "pinned "
#endif
#endif //CUDA

	 << "| termination: "
#ifdef AVG_TERMINATION
	 << "average "
#else
	 << "max "
#endif
	 << endl

	 << "control-window size: " << w_max << endl
	 << "alpha = " << alfa << "; beta = " << beta << endl
	 << "number of workers = " << nworkers << endl
	 << "max number of cycles = " << max_cycles << endl;
    if(fixed_cycles)
      cout << "number of cycles fixed to " << max_cycles <<  endl;
    cout << "Picture: " << fname << endl
	 << "Restored picture: " << prefix << ".bmp" << endl << "---" << endl;
  }

  



  /*
    ---------------
    READ the bitmap
    ---------------
  */
  if(verbose)
    cout << "reading the original picture... " << flush;

  Bitmap<grayscale> bmp;
  bmp.read_8bit_nopalette(fp);
  bmp_size_t width = bmp.width(), height = bmp.height();

  if(verbose)
    cout << "ok [W = " << width << ", H = " << height << "]" << endl;





  /*
    -------------------------
    DETECT noisy pixels
    -------------------------
  */
  if(verbose)
    cout << "searching for noisy pixels... " << flush;
#ifdef TIME
  t_finding = get_usec_from(0);
#endif

  Bitmap_ctrl<grayscale> bmp_ctrl(bmp);
  bmp_size_t n_noisy = 0;



#ifndef SEQ_DETECTION
  //PARALLEL detection
  if(verbose)
    cout << "(parallel) ";
  unsigned int n_detectors = nworkers;
  vector<vector<noisy<grayscale> > > noisy_sets_(n_detectors);
  //set the farm
  ff::ff_farm<> farm_detection(true); //accelerator set
  std::vector<ff::ff_node *> w_detectors;
  for(unsigned int i=0; i<n_detectors; ++i)
    w_detectors.push_back(new Worker_detector<grayscale>(bmp, bmp_ctrl, width, noisy_sets_, w_max));
  farm_detection.add_workers(w_detectors);
 
  //detect
  bmp_size_t step_rows = height / n_detectors;
  bmp_size_t first_row = 0;
  bmp_size_t last_row;
  unsigned int n_noisy_sets_ = 0;
  farm_detection.run();
  while(true) {
    last_row = min(first_row + step_rows, height - 1);
    //cout << first_row << " -> " << last_row << "\n";
    farm_detection.offload(new detection_task_t(first_row, last_row, n_noisy_sets_++));
    if(last_row == (height - 1))
      break;
    first_row = last_row + 1;
  }

  //join
  farm_detection.offload((void *)ff::FF_EOS);
  farm_detection.wait();

  //compute n_noisy and (if required) fold
#ifdef CUDA
  vector<noisy<grayscale> > noisy_pixels_wrap;
  vector<noisy<grayscale> >::iterator it = noisy_pixels_wrap.begin();
#endif
  for(unsigned int i=0; i<n_noisy_sets_; ++i) {
    n_noisy += noisy_sets_[i].size();
#ifdef CUDA
    /*
    cerr << "will copy set " << i << "/" << noisy_sets_.size() << endl;
    //noisy_pixels_wrap.reserve(n_noisy);
    noisy_pixels_wrap.insert(it, noisy_sets_[i].begin(), noisy_sets_[i].end());
    it += noisy_sets_[i].size();
    cerr << "copied set " << i << "/" << noisy_sets_.size() << endl;
    */
    for(unsigned int j=0; j<noisy_sets_[i].size(); ++j)
      noisy_pixels_wrap.push_back(noisy_sets_[i][j]);
#endif
  }



#else
  //SEQUENTIAL detection
  if(verbose)
    cout << "(sequential) ";
  vector<noisy<grayscale> > noisy_pixels_wrap;
  noisy_pixels_wrap.reserve(height * width);
  n_noisy = find_noisy_partial<grayscale>(bmp, bmp_ctrl, w_max, noisy_pixels_wrap, 0, height - 1, width);
  noisy<grayscale> *noisy_pixels = (noisy<grayscale> *)malloc(n_noisy * sizeof(noisy<grayscale>));
  for(unsigned int i=0; i<n_noisy; ++i)
    noisy_pixels[i] = noisy_pixels_wrap[i];
#endif



  if(verbose)
    cout << "ok [" << n_noisy << " noisy pixels]";
#ifdef TIME
  t_finding = get_usec_from(t_finding)/1000;
  if(verbose)
    cout << " (" << t_finding << " ms)";
#endif
  if(verbose)
    cout << endl;





#ifndef NO_BLOCKS
  /*
    --------------------------
    BUILD noisy blocks
    --------------------------
  */
  if(verbose)
    cout << "clustering noisy pixels... " << flush;
#ifdef TIME
  t_clustering = get_usec_from(0);
#endif


  //select clustering strategy
#ifndef CLUSTER
  //default strategy: BLOCK
#ifdef CONTINOUS
  unsigned int n_blocks = 2 * nworkers;
#else //CONTINUOUS
  unsigned int n_blocks = K_BLOCKS * nworkers;
#endif //CONTINUOUS
  vector<vector<noisy<grayscale> > > noisy_sets(n_blocks);
  build_blocks<noisy<grayscale> >(noisy_sets, noisy_sets_, n_noisy, n_blocks);
 
#ifdef BORDER
  //find borders (for each block)
  vector<pair<unsigned int, unsigned int> > borders;
  find_borders<grayscale>(borders, noisy_sets, n_blocks, height, width, bmp);
#endif //BORDER

#else //CLUSTER
  vector<vector<noisy<grayscale> > > noisy_sets;
  noisy_sets.reserve(1000);
  build_perfect_clusters<noisy<grayscale> >(noisy_sets, noisy_pixels, n_noisy);
#endif //CLUSTER


  if(verbose)
    cout << "ok [" << noisy_sets.size() << " noisy sets]";
#ifdef TIME
  t_clustering = get_usec_from(t_clustering)/1000;
  if(verbose)
    cout << " (" << t_clustering << " ms)";
#endif
  if(verbose)
    cout << endl;
#endif //NO_BLOCKS





  /*
    -----------------
    DENOISE (cycles)
    -----------------
  */
  if(verbose)
    cout << "denoising... " << flush;
#ifdef TIME
  t_correction = get_usec_from(0);
#endif



#ifndef NO_BLOCKS
  //structures for convergence check
  residual_set sets_residuals(noisy_sets.size(), 0); //setwise residuals
  vector<vector<grayscale> > diff(noisy_sets.size()); //pixelwise diff
  
  for (unsigned int i=0; i< noisy_sets.size(); ++i) {
    vector<grayscale> t(noisy_sets[i].size(), (grayscale)0);
    diff[i] = t;
  }
#endif
  unsigned int cycle = 0; //number of completed cycles


#ifndef CONTINOUS
  //not CONTINUOUS scheme
  float current_residual, old_residual, delta_residual;

#ifdef CUDA
  //CUDA denoiser
  Cuda_denoiser<grayscale> cuda_denoiser(&bmp, alfa, beta, noisy_pixels_wrap, height, width, verbose);
  cuda_denoiser.svc_init();

#else
  //FF-accelerator denoiser
  ff::ff_farm<> farm(true); //accelerator set
  std::vector<ff::ff_node *> w;
  for(unsigned int i=0; i<nworkers; ++i)
    w.push_back(new FFW_Filter<grayscale>(bmp, alfa, beta, width, height, noisy_sets, diff, sets_residuals));
  farm.add_workers(w);
#endif

  //start
  float epsilon_residual = EPS_STOP;
  current_residual = 0;
#ifdef CONV_MINIMUM
  float min_residual = numeric_limits<residual_t>::max();
  bool minimum = false;
#endif
  bool fix = false;
  while(!fix) {

    //backup
    old_residual = current_residual;
    
#ifdef FLAT
    //backup ALL noisy pixels
    /* seq. version
    for(unsigned int i=0; i<noisy_sets.size(); i++)
      for(unsigned int j=0; j<noisy_sets[i].size(); ++j)
	bmp.backup(noisy_sets[i][j].c, noisy_sets[i][j].r);
    */
    farm.run_then_freeze();
    for(unsigned int j=1; j<=noisy_sets.size(); j++)
      if (noisy_sets[j-1].size() > 0)
	farm.offload((void *)(j + noisy_sets.size()));
    farm.offload((void *)ff::FF_EOS);
    farm.wait_freezing();

#elif defined(BORDER)
    //backup the borders
    for(unsigned int k=0; k<borders.size(); k++) {
	  int i = borders[k].first;
	  int j = borders[k].second;
	  bmp.backup(noisy_sets[i][j].c, noisy_sets[i][j].r);
	}
#endif



#ifdef CUDA
    //CUDA restoring pass
    current_residual = cuda_denoiser.svc();

#else //CUDA
     //FF-accelerator restoring pass
    farm.run_then_freeze();
    for(unsigned int j=1; j<=noisy_sets.size(); j++)
      if (noisy_sets[j-1].size() > 0)
	farm.offload((void *)j);
    farm.offload((void *)ff::FF_EOS);
    farm.wait_freezing();

#ifdef AVG_TERMINATION
    current_residual = residual_avg<grayscale>(sets_residuals, diff, n_noisy);
#else
    current_residual = residual_max<grayscale>(sets_residuals, n_noisy);
#endif
#endif //CUDA
    //cout << "current residual " << current_residual << endl;

    //check convergence
#ifdef CONV_MINIMUM
    if(current_residual < min_residual)
      min_residual = current_residual;
    else
      if(current_residual == min_residual) {
	if(verbose)
	  cout << "residual stall" << endl;
      }
      else {
	if(verbose)
	  cout << "local minimum detected" << endl;
	minimum = true;
      }
#endif
    
    delta_residual = _ABS((old_residual - current_residual));

    //fixed point?
    ++cycle;
    if(fixed_cycles)
      fix = (cycle == max_cycles);
    else {
      fix = (delta_residual < epsilon_residual || cycle == max_cycles);
#ifdef CONV_MINIMUM
      fix = fix || minimum;
#endif
    }

#ifdef ALTERNATE
    //invert mode
    for(unsigned int i=0; i<nworkers; i++) {
      Worker *cw = static_cast<Worker *>(w[i]);
      cw->invert_mode();
    }
#endif

#ifdef TIME
    //t_red += get_usec_from(t_redp);
#endif
    //cout << "Fine ciclo n. " << (i+1) << " Max residual "<< (int) res << endl;
    /*
      #ifdef TIME
      usec_tmp = get_usec_from(usec_tmp);	
      cout << "Riduzione ciclo n. " << (i+1) << " time " << usec_tmp/1000 << "ms\n";
      #endif
      */

#ifdef WRITE_PASSES
    char prefixstep[15];
    sprintf(prefixstep,"restored_pass%d_",(cycle+1));
    bmp_finalizer(fp, bmp, width, height, prefixstep);
#endif
  }

#ifdef FF_ACCEL
  //join
  farm.offload((void *)ff::FF_EOS);
  farm.wait();
#elif defined(CUDA)
  cuda_denoiser.svc_end();
#endif



#else // CONTINOUS
  ff::ff_farm<> farm(false); //accelerator not set
  std::vector<ff::ff_node *> w;
  for(unsigned int i=0; i<nworkers; ++i)
    w.push_back(new FFW_Filter<grayscale>(bmp, alfa, beta, width, height, noisy_sets, diff, sets_residuals));
  farm.add_workers(w);
  farm.add_emitter(new FFE_EmitConv<grayscale>(noisy_sets.size(), EPS_STOP, n_noisy, diff, sets_residuals, cycle, max_cycles)); 
  farm.wrap_around();
  farm.run_and_wait_end();
#endif // CONTINOUS



if(verbose)
    cout << "ok [" << cycle << " cycles]";

#ifdef PASSES
  if(!verbose)
    cout << cycle << endl;
#endif
  
#ifdef TIME
  t_correction = get_usec_from(t_correction)/1000;
  if(verbose)
    cout << " (" << t_correction << " ms)";
#endif
  if(verbose)
    cout << endl;




  /*
    ---------------------
    WRITE restored bitmap
    ---------------------
  */
  if(verbose)
    cout << "writing the restored picture... " << flush;

  write_8bit_nopalette(fp, bmp, width, height, prefix);

  //close files and clean-up
  fclose(fp);
  if(verbose)
    cout << "ok" << endl;
#ifdef CLUSTER
  free(noisy_pixels);
#endif

#ifdef TIME
  t_whole = get_usec_from(t_whole)/1000;
  if(verbose) {
    cout << "total time: " << t_whole << " ms" << endl;
    //cout << "reduce time: " << t_red/1000.0 << " ms" << endl;
  }
  else
    cout << double(t_whole)/1000 << endl;
#endif
  return 0;
}
