#ifndef RBM_SHARED_UNITS_H
#define RBM_SHARED_UNITS_H

#ifndef OCL_KERNEL

#include <math.h>
#include <stdint.h>
#include <string>
#include <map>

namespace ailab {

#ifndef decimal_t
typedef float decimal_t;
#endif

// Any code which is required on the CPU and NOT in OpenCL

// Fake these OpenCL namespace specifiers
#define __global
#define __local
#define __constant const

#define clamp(x, minval, maxval) std::fmin(std::fmax(x, minval), std::maxval)

typedef struct _uint4_t_ {
  uint32_t x;
  uint32_t y;
  uint32_t z;
  uint32_t w;

  void operator=(const _uint4_t_& rhs) {
    this->x = rhs.x;
    this->y = rhs.y;
    this->z = rhs.z;
    this->w = rhs.w;
  }
  void operator=(const uint32_t& rhs) {
    this->x = rhs;
    this->y = rhs;
    this->z = rhs;
    this->w = rhs;
  }
} uint4;

typedef struct {
  uint32_t x;
  uint32_t y;
} uint2;

typedef struct {
  decimal_t x;
  decimal_t y;
} decimal2;

typedef struct {
  decimal_t x;
  decimal_t y;
  decimal_t z;
  decimal_t w;
} decimal4;

typedef struct {
  decimal_t s0;
  decimal_t s1;
  decimal_t s2;
  decimal_t s3;
  decimal_t s4;
  decimal_t s5;
  decimal_t s6;
  decimal_t s7;
} decimal8;

typedef decimal2 rbm_unit_state_t;
typedef decimal4 rbm_unit_params_t;

typedef decimal_t (*RBMApplyFunc)(decimal_t total_input, decimal_t bias,
    __global rbm_unit_state_t * state,
    __global rbm_unit_params_t * params,
    __constant decimal_t * typeParams);

typedef struct _rbm_shared_func {
  const char* name;
  RBMApplyFunc func;
} RBMSharedFunc;

typedef struct {
  const char * name;
  bool can_be_sampled;
  int param_count;
  int unit_param_count;
  RBMApplyFunc activation;
  RBMApplyFunc inner_sum;
} UnitConfig;

typedef std::map<std::string, UnitConfig> UnitCfgMap;
extern UnitCfgMap global_rbm_all_units_config;

decimal_t select(decimal_t a, decimal_t b, unsigned char c);
decimal_t fclamp( decimal_t x, decimal_t minval, decimal_t maxval);

#endif

/*-------------------- Begin NVIDIA Code ---------------------------*/
// Code for Gaussian RNG from GPUGems 3, slightly modified for OpenCL
// see: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
uint32_t tausworthe(uint32_t *z, int a, int b, int c, uint32_t m);
uint32_t LCGStep(uint32_t *z, uint32_t a, uint32_t c);

decimal_t uniformRNG(uint4 * s);
decimal2 gaussianRNG(uint4 * state);
/* --------------------- End of NVIDIA Code -------------------------------*/

/* ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^ */
/* ----------------------------- Handle unit input sum ----------------------------- */

decimal_t rbm_weighted_inner_sum(decimal_t u, decimal_t w,
                                 __global rbm_unit_state_t * state,
                                 __global rbm_unit_params_t * params,
                                 __constant decimal_t * typeParams);

// If you need (u*w) / params, just set params to 1/params; that will be much faster than
// using division here
decimal_t rbm_scaled_inner_sum(decimal_t u, decimal_t w,
__constant decimal_t * params);

/* ----------------------------- Unit Output Functions ----------------------------- */
decimal_t rbm_rectified_linear_unit(decimal_t total_input, decimal_t bias,
                                    __global rbm_unit_state_t * state,
                                    __global rbm_unit_params_t * params,
                                    __constant decimal_t * typeParams);

decimal_t rbm_binary_unit(decimal_t total_input, decimal_t bias,
                            __global rbm_unit_state_t * state,
                            __global rbm_unit_params_t * params,
                            __constant decimal_t * typeParams);


decimal_t rbm_gaussian_unit(decimal_t total_input, decimal_t bias,
                            __global rbm_unit_state_t * state,
                            __global rbm_unit_params_t * params,
                            __constant decimal_t * typeParams);

decimal_t rbm_simple_neuron_unit(decimal_t total_input, decimal_t bias,
                                 __global rbm_unit_state_t * state,
                                 __global rbm_unit_params_t * params,
                                 __constant decimal_t * typeParams);

/* ----------------------------- Unit energy_functions ----------------------------- */
decimal_t rbm_basic_unit_energy(decimal_t activity, decimal_t bias,
                                __global rbm_unit_state_t * state,
                                __global rbm_unit_params_t * params,
                                __constant decimal_t * typeParams);

decimal_t rbm_gaussian_unit_energy(decimal_t activity, decimal_t bias,
                                   __global rbm_unit_state_t * state,
                                   __global rbm_unit_params_t * params,
                                   __constant decimal_t * typeParams);

/* ----------------------------- Sampling funcitons ----------------------------- */
decimal_t rbm_binary_sample(decimal_t bernoulli_value, uint4 * rng_state,
__constant decimal_t * params);
decimal_t rbm_binomial_sample(decimal_t bernoulli_value, uint4 * rng_state,
__constant decimal_t * params);

/* ----------------------------- Joint Sampling funcitons ----------------------------- */
void rbm_joint_binary_sample(uint32_t start, uint32_t end,
                             __global decimal_t * activity, uint4 * rng_state,
                             __constant decimal_t * params);
void rbm_joint_normalize(uint32_t start, uint32_t end,
                         __global decimal_t * activity, uint4 * rng_state,
                         __constant decimal_t * params);
/* ----------------------------- Unit Entropy Functions ----------------------------- */
decimal_t rbm_binary_entropy(decimal_t val, decimal_t bias,
__constant decimal_t * params);
decimal_t rbm_gaussian_unit_variance_entropy(decimal_t val, decimal_t bias,
__constant decimal_t * params);
decimal_t rbm_gaussian_entropy(decimal_t val, decimal_t bias,
__constant decimal_t * params);

/* ----------------------------- Probability Distribution Functions ----------------------------- */
decimal_t rbm_binary_pdf(decimal_t value, decimal_t bias,
__constant decimal_t * params);

#ifndef OCL_KERNEL
}
#endif

#endif // RBM_SHARED_UNITS_H
