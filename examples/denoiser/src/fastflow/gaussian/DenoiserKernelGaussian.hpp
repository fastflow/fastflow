#include <Denoiser.hpp>
#include <fastflow/gaussian/kernel_cpp/fuyGauss.hpp>

struct DenoiserKernelGaussian_params {
  float alfa, beta;
};

/*!
 * \class DenoiserGaussian
 *
 * \brief Restored a frame affected by gaussian noise
 *
 * This class restores a frame affected by gaussian noise
 *
 */
class DenoiserKernelGaussian {
public:
  /**
   *
   * @param height frame height
   * @param width frame h
   * @param alfa_ functional parameter alfa
   * @param beta_ functional parameter beta
   * @param fixed_cycles flag for variable/fixed number of cycles
   * @param max_cycles maximum number of cycles
   * @param trace_time_ tells if time should be measured
   */
  DenoiserKernelGaussian(void *params_, unsigned int height_, unsigned int width_) :
      height(height_), width(width_) {
    DenoiserKernelGaussian_params *params = (DenoiserKernelGaussian_params *)params_;
    alfa = params->alfa;
    beta = params->beta;
  }
  /**
   *
   * @param in input frame
   * @param x pixel index
   * @param noisymap noisy pixels map
   * @return
   */
  unsigned char operator()(unsigned char *in, unsigned int x, int *noisymap) {
    return fuy(in, x, noisymap, width, height, alfa, beta);
  }

  void restore_chunk(unsigned int *noisy, unsigned int first, unsigned int last, int *noisymap, unsigned char *in, unsigned char *out) {
    for (int i=first; i<=last; ++i)
      {
	unsigned int x = noisy[i];
        out[x] = fuy(in, x, noisymap, width, height, alfa, beta);
      }
  }

private:
  float alfa, beta;
  unsigned int height, width;
};
