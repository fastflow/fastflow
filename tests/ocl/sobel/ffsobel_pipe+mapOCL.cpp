/* -*- Mode: C++; tab-width: 4; c-basic-offset: 4; indent-tabs-mode: nil -*- */
/* ***************************************************************************
 *  This program is free software; you can redistribute it and/or modify
 *  it under the terms of the GNU General Public License version 2 as 
 *  published by the Free Software Foundation.
 *
 *  This program is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU General Public License for more details.
 *
 *  You should have received a copy of the GNU General Public License
 *  along with this program; if not, write to the Free Software
 *  Foundation, Inc., 59 Temple Place - Suite 330, Boston, MA 02111-1307, USA.
 *
 *  As a special exception, you may use this file as part of a free software
 *  library without restriction.  Specifically, if other files instantiate
 *  templates or use macros or inline functions from this file, or you compile
 *  this file and link it with other files to produce an executable, this
 *  file does not by itself cause the resulting executable to be covered by
 *  the GNU General Public License.  This exception does not however
 *  invalidate any other reasons why the executable file might be covered by
 *  the GNU General Public License.
 *
 ****************************************************************************
 */

/*  Computes the Sobel filter on each input image. For information concerning 
 *  the Sobel filter please refer to :
 *   http://en.wikipedia.org/wiki/Sobel_operator
 * 
 *  
 *  command: 
 *
 *    ffsobel_pipe+mapOCL [-r repeatTime]  file1 file2 file3 ....
 *
 *  the output is produced in the directory ./out
 *
 */
/* 
 * Author: Massimo Torquati <torquati@di.unipi.it> 
 * Date:   October 2014
 *         August  2015 (run-time selection between CPU and GPU computation)
 */

#if !defined(FF_OPENCL)
#define FF_OPENCL
#endif

#include<iostream>
#include<cmath>
#include<opencv2/imgproc/imgproc.hpp>
#include<opencv2/highgui/highgui.hpp>

#include <ff/pipeline.hpp>
#include <ff/stencilReduceOCL.hpp>
#include <ff/map.hpp>
#include <ff/selector.hpp>

using namespace cv;
using namespace ff;

const long NUM_WRITERS     = 8; // number of threads writing data into the disk
const long NUM_CPU_WORKERS = 4; // parallelism degree for the CPU map (they computes in parallel small images)
const long SMALL_SIZE     = 250000; // we consider an image small if #cols*rows <= SMALL_SIZE
long repeatTime           = 1; // to repeat the same set of images a numer of times



/* --------------- OpenCL code ------------------- */
FF_OCL_STENCIL_ELEMFUNC_ENV(mapf, uchar, useless, k, usrc, env_t, env,
               (void)useless;
               long cols = env->cols;
               long rows = env->rows;                  
               long y    = k / cols; if (y==0 || y==(cols-1)) return 0;
               long x    = k % cols; if (x==0 || x==(cols-1)) return 0;
               
               long gx = xGradient(usrc, cols, x, y);
               long gy = yGradient(usrc, cols, x, y);
               long sum = abs(gx) + abs(gy); 
               if (sum > 255) sum = 255;
               else if (sum < 0) sum = 0;
               
               return sum;
);

/* ----- utility function ------- */
// helping functions: it parses the command line options
const char* getOption(char **begin, char **end, const std::string &option) {
    char **itr = std::find(begin, end, option);
    if (itr != end && ++itr != end) return *itr;
    return nullptr;
}

template<typename T>
T *Mat2uchar(cv::Mat &in) {
    T *out = new T[in.rows * in.cols];
    for (int i = 0; i < in.rows; ++i)
        for (int j = 0; j < in.cols; ++j)
            out[i * (in.cols) + j] = in.at<T>(i, j);
    return out;
}
#define XY2I(Y,X,COLS) (((Y) * (COLS)) + (X))
/* ------------------------------ */



// returns the gradient in the x direction
static inline long xGradient(uchar * image, long cols, long x, long y) {
    return image[XY2I(y-1, x-1, cols)] +
        2*image[XY2I(y, x-1, cols)] +
        image[XY2I(y+1, x-1, cols)] -
        image[XY2I(y-1, x+1, cols)] -
        2*image[XY2I(y, x+1, cols)] -
        image[XY2I(y+1, x+1, cols)];
}

// returns the gradient in the y direction 
static inline long yGradient(uchar * image, long cols, long x, long y) {
    return image[XY2I(y-1, x-1, cols)] +
        2*image[XY2I(y-1, x, cols)] +
        image[XY2I(y-1, x+1, cols)] -
        image[XY2I(y+1, x-1, cols)] -
        2*image[XY2I(y+1, x, cols)] -
        image[XY2I(y+1, x+1, cols)];
}


// the corresponding OpenCL type is in the (local) file 'ff_opencl_datatypes.cl'
struct env_t {
    env_t() : cols(0), rows(0) {}
    env_t(long cols, long rows):cols(cols),rows(rows) {}
    long   cols;
    long   rows;
};

/* This is data flowing between pipeline stages */
struct Task {
    Task(uchar *src, uchar *dst, long rows, long cols, const std::string &name):
        src(src),dst(dst),env(cols,rows), name(name) {};
    uchar             *src, *dst;
    env_t              env; // contains 'rows' and 'cols'
    const std::string  name;
};

/* this is the task used in the OpenCL map 
 * 
 */
struct oclTask: public baseOCLTask<Task, uchar, uchar> {
    oclTask() {}
    void setTask(Task *t) { 
        assert(t);
        size_t N = t->env.rows * t->env.cols;
        setInPtr(t->src, N);
        setOutPtr(t->dst, N);

        setEnvPtr(&t->env, 1);
    }
};

// it dynamically selects the logical device where the kernel has to be executed.
//  0 for this test is the C++ Map (based on the parallel_for pattern)
//  1 for this test is the OpenCL Map
struct Kernel: ff_nodeSelector<Task> {
    Kernel(const int preferredDevice = -1):preferredDevice(preferredDevice) {}

    Task *svc(Task *in) {
       
        const long cols = in->env.cols;
        const long rows = in->env.rows;

        int selectedDevice = preferredDevice;
        if (selectedDevice < 0) {
            if (cols*rows > SMALL_SIZE) selectedDevice = 1;
            else selectedDevice = 0;
        }             
        selectNode(selectedDevice);

        unsigned long x = getusec();
        in = ff_nodeSelector<Task>::svc(in);
        unsigned long y = getusec();
        printf("(%ld,%ld) %ld (us) selected=%s\n", 
               in->env.rows, in->env.cols, y-x, selectedDevice?"GPU":"CPU");

        return in;            
    }
private:
    int preferredDevice;
};

struct cpuMap: ff_Map<Task,Task> {    
    // sets the maximum n. of worker for the Map
    // spinWait is set to true
    // the scheduler is disabled by default
    cpuMap(const long mapworkers=std::max(ff_realNumCores(),NUM_CPU_WORKERS)):ff_Map<Task,Task>(mapworkers) {} //,true) {}
    
    Task *svc(Task *task) {
        uchar * src = task->src, * dst = task->dst;
        const long cols   =  task->env.cols;
        const long rows   =  task->env.rows;
        ff_Map<Task,Task>::parallel_for(1,rows-1,[src,cols,&dst](const long y) {
                for(long x = 1; x < cols - 1; x++){
                    const long gx = xGradient(src, cols, x, y);
                    const long gy = yGradient(src, cols, x, y);
                    // approximation of sqrt(gx*gx+gy*gy)
                    long sum = abs(gx) + abs(gy); 
                    if (sum > 255) sum = 255;
                    else if (sum < 0) sum = 0;
                    dst[y*cols+x] = sum;
                }
            }, (cols*rows>SMALL_SIZE)?ff_realNumCores():NUM_CPU_WORKERS); 
        return task;
    }
};



int main(int argc, char *argv[]) {
    if (argc < 2) {
        std::cerr << "use: " << argv[0] 
		  << " [-r num_repeat] [-g cpu-or-gpu (-1|0|1)] <image-file> [image-file]\n";
        return -1;
    }    

    int cpuorgpu = -1;  // by default selects the device on the base of the image size
    int start    =  1;
    long num_images = argc-1;

    const char *r = getOption(argv, argv+argc, "-r");
    if (r) { repeatTime = atol(r); start+=2; num_images -= 2; }
    assert(num_images >= 1);
    const char *g = getOption(argv, argv+argc, "-g");
    if (g) { cpuorgpu = atoi(g); start+=2; num_images -= 2; }

    printf("num_images= %ld\n", num_images);
    if  (cpuorgpu<0) printf("selecting the device dynamically\n");
    else             printf("executing %s\n", cpuorgpu?"OpenCL Map":"CPU Map");

    auto reader= [&](Task*, ff_node*const stage) -> Task* {
        
        for(long k=0;k<repeatTime;++k) {    // repeating a number of times the same set of images
            for(long i=0; i<num_images; ++i) {
                const std::string &filepath(argv[i+start]);
                std::string filename;
                
                // get only the filename
                int n=filepath.find_last_of("/");
                if (n>0) filename = filepath.substr(n+1);
                else     filename = filepath;
                
                Mat * src = new Mat;
                *src = imread(filepath, CV_LOAD_IMAGE_GRAYSCALE);
                if ( !src->data ) {
                    std::cerr << "ERROR reading image file " << filepath << " going on....\n";
                    continue;
                }
                const long cols = src->cols;
                const long rows = src->rows;
                uchar * dst = new uchar[rows * cols]; 
                for(long y = 0; y < rows; y++)
                    for(long x = 0; x < cols; x++) {
                        dst[y * cols + x] = 0;
                    }        
                
                Task *t = new Task(Mat2uchar<uchar>(*src), dst, rows, cols, filename);
                stage->ff_send_out(t); // sends the task t to the next stage
                delete src;
            }
        }
        return static_cast<Task*>(EOS);
    };

    auto writer= [](Task* task, ff_node*const) -> Task*  {
        uchar * usrc = task->src, * dst = task->dst;
        long   cols = task->env.cols, rows = task->env.rows;
    
        const std::string &outfile = "./out/" + task->name;
        imwrite(outfile,  cv::Mat(rows, cols, CV_8U, dst, cv::Mat::AUTO_STEP));
        //std::cout << "File " << task->name << " has been written into the out dir\n";
        delete [] dst;
        delete [] usrc; 
        delete    task;
        return static_cast<Task*>(GO_ON);
    };

    // OpenCL map
    ff_mapOCL_1D<Task, oclTask> sobel(mapf);
    // CPU map
    cpuMap cpusobel;

    ff_node_F<Task> Reader(reader);
    Kernel kern(cpuorgpu);
    kern.addNode(cpusobel);
    kern.addNode(sobel);

    // here we use a farm to write in parallel multiple images
    ff_Farm<Task> farm(writer, NUM_WRITERS); 
    farm.remove_collector();
    ff_Pipe<> pipe(Reader, kern, farm);

    ffTime(START_TIME);
    if (pipe.run_and_wait_end()<0) {
        error("running pipeline\n");
        return -1;
    }        
    ffTime(STOP_TIME);
    printf("Time= %g (ms)\n", ffTime(GET_TIME));

    return 0;  
}
