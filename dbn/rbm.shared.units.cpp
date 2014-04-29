#ifdef OCL_KERNEL
// Any code which is required in OpenCL and NOT the CPU
    decimal_t fclamp( decimal_t x, decimal_t minval, decimal_t maxval) {
      return clamp(x, minval, maxval);
    }

#else

// Any code which is required on the CPU and NOT in OpenCL
#include <ailab/algorithms/rbm.shared.units.h>
#include <stdint.h>
#include <cstddef>
#include <assert.h>
#include <iostream>

namespace ailab {

  decimal_t select(decimal_t a, decimal_t b, unsigned char c){
    return c ? b : a;
  }

  decimal_t fclamp( decimal_t x, decimal_t minval, decimal_t maxval) {
    return (x > maxval) ? maxval : ((x < minval) ? minval : x);
  }

#endif

// GENERAL Definitions
#define DECIMAL_PARAM(x) (*((decimal_t *)x))
#define UINT_PARAM(x) (*((uint32_t *)x))
#define INT_PARAM(x) (*((int32_t *)x))

#define TWO_PI 6.283185307179586f

/*-------------------- Begin NVIDIA Code ---------------------------*/
// Code for Gaussian RNG from GPUGems 3, slightly modified for OpenCL
// see: http://http.developer.nvidia.com/GPUGems3/gpugems3_ch37.html
uint32_t tausworthe(uint32_t *z, int a, int b, int c, uint32_t m) {
  uint32_t t = ((((*z) << a) ^ (*z)) >> b);
  return *z = ((((*z) & m) << c) ^ t);
}

uint32_t LCGStep(uint32_t *z, uint32_t a, uint32_t c) {
  return *z = (a * (*z) + c);
}

decimal_t uniformRNG(uint4 * s) {
  // Combined period is lcm(p1,p2,p3,p4)~ 2^121
  // Periods
  uint32_t a = s->x;
  uint32_t b = s->y;
  uint32_t c = s->z;
  uint32_t d = s->w;

  decimal_t num = 2.3283064365387e-10f
      * (tausworthe(&a, 13, 19, 12, 4294967294UL) ^  // p1=2^31-1
          tausworthe(&b, 2, 25, 4, 4294967288UL) ^  // p2=2^30-1
          tausworthe(&c, 3, 11, 17, 4294967280UL) ^  // p3=2^28-1
          LCGStep(&d, 1664525, 1013904223UL)      // p4=2^32
      );
  s->x = a;
  s->y = b;
  s->z = c;
  s->w = d;
  return num;
}

/* --------------------- End of NVIDIA Code -------------------------------*/

/* ----------------------------- Handle unit input sum ----------------------------- */

decimal_t rbm_weighted_inner_sum(decimal_t u, decimal_t w,
                                 __global rbm_unit_state_t * state,
                                 __global rbm_unit_params_t * params,
                                 __constant decimal_t * typeParams) {
  return u * w;
}

// If you need (u*w) / params, just set params to 1/params; that will be much faster than
// using division here
decimal_t rbm_scaled_inner_sum(decimal_t u, decimal_t w,
                               __global rbm_unit_state_t * state,
                               __global rbm_unit_params_t * params,
                               __constant decimal_t * typeParams) {
  return u * w * (*((decimal_t*) typeParams));
}

/* ----------------------------- Unit Activation Functions ----------------------------- */

/***
 Your output function must have a finite range.
 Without the entire network may saturate with NaN / Infinity values;
 Imposing limits down stream is a bit too limiting
 ***/
decimal_t rbm_rectified_linear_unit(decimal_t total_input, decimal_t bias,
                                    __global rbm_unit_state_t * state,
                                    __global rbm_unit_params_t * params,
                                    __constant decimal_t * typeParams) {
  uint32_t iters = *((uint32_t *) typeParams);
  decimal_t sum = 0;
  for (uint32_t i = 1; i < iters; i++) {
    sum += 1.0f / (1.0f + exp(-(total_input + 0.5f - i)));
  }
  return sum;
}

decimal_t rbm_rectified_linear_log_unit(decimal_t total_input, decimal_t bias,
                                        __global rbm_unit_state_t * state,
                                        __global rbm_unit_params_t * params,
                                        __constant decimal_t * typeParams) {
  return log(1.0f + exp(-(bias + total_input)));
}

decimal_t rbm_binary_unit(decimal_t total_input, decimal_t bias,
                          __global rbm_unit_state_t * state, __global rbm_unit_params_t * params,
                          __constant decimal_t * typeParams) {
  return (1.0f / (1.0f + exp(-(total_input + bias))));
}

decimal_t rbm_linear_unit(decimal_t total_input, decimal_t bias,
                          __global rbm_unit_state_t * state, __global rbm_unit_params_t * params,
                          __constant decimal_t * typeParams) {
  return total_input + bias;
}

decimal_t rbm_gaussian_unit(decimal_t total_input, decimal_t bias,
                            __global rbm_unit_state_t * state,
                            __global rbm_unit_params_t * params,
                            __constant decimal_t * typeParams) {
  return (total_input * DECIMAL_PARAM(params)) + bias;
}

/**
 * Using Spiking Model by Izhikevich
 * Citation: Izhikevich, Eugene M. "Simple model of spiking neurons." Neural Networks, IEEE Transactions on 14, no. 6 (2003): 1569-1572.
 */
decimal_t rbm_simple_neuron_unit(decimal_t total_input, decimal_t bias,
                                 __global rbm_unit_state_t * state,
                                 __global rbm_unit_params_t * params,
                                 __constant decimal_t * typeParams) {
  decimal_t v = state->x;
  decimal_t u = state->y;
  decimal_t p1 = typeParams[0];
  decimal_t p2 = typeParams[1];
  decimal_t p3 = typeParams[2];
  decimal_t threshold = typeParams[3];

  decimal_t vprime = (p1 * v * v) + (p2 * v) + (p3) - u + total_input;
  decimal_t uprime = params->x * ((params->y * v) - u);

  uprime = select(u, u + params->z,  isgreater(vprime, threshold));
  vprime = select(vprime, params->y, isgreater(vprime, threshold));

  state->x = vprime;
  state->y = uprime;

  return fclamp(vprime, 0.0, 1.0);
}

/* ----------------------------- Unit energy_functions ----------------------------- */
decimal_t rbm_basic_unit_energy(decimal_t activity, decimal_t bias,
                                __global rbm_unit_state_t * state,
                                __global rbm_unit_params_t * params,
                                __constant decimal_t * typeParams) {
return activity * bias;
}

decimal_t rbm_gaussian_unit_energy(decimal_t activity, decimal_t bias,
                                   __global rbm_unit_state_t * state,
                                   __global rbm_unit_params_t * params,
                                   __constant decimal_t * typeParams) {
return pow(activity - bias, 2) / (2 * pow(DECIMAL_PARAM(typeParams), 2));
}

/* ----------------------------- Sampling funcitons ----------------------------- */
decimal_t rbm_binary_sample(decimal_t p_on, uint4 * rng_state,
__constant decimal_t * typeParams) {
  return 1.0f * (p_on > uniformRNG(rng_state));
}

decimal_t rbm_binomial_sample(decimal_t p_on, uint4 * rng_state,
__constant decimal_t * typeParams) {
  decimal_t r = 0;
  for (uint32_t i = 0; i < UINT_PARAM(typeParams); i++)
    r += 1.0f * (p_on > uniformRNG(rng_state));
  return r;
}

/* ----------------------------- Unit Entropy Functions ----------------------------- */
decimal_t rbm_binary_entropy(decimal_t val, decimal_t bias,
                             __global rbm_unit_state_t * state,
                             __global rbm_unit_params_t * params,
                             __constant decimal_t * typeParams) {
  decimal_t p1 = bias;
  decimal_t p0 = 1 - bias;

  return -((p0 * log(p0)) + (p1 * log(p1)));
}

decimal_t rbm_gaussian_unit_variance_entropy(decimal_t val, decimal_t bias,
                                             __global rbm_unit_state_t * state,
                                             __global rbm_unit_params_t * params,
                                             __constant decimal_t * typeParams) {
  return log(TWO_PI * (M_E)) / 2.0f;
}

/* ----------------------------- Probability Distribution Functions ----------------------------- */
decimal_t rbm_binary_pdf(decimal_t value, decimal_t bias,
                         __global rbm_unit_state_t * state,
                         __global rbm_unit_params_t * params,
                         __constant decimal_t * typeParams) {
return fmax((1.0f - value) * (1.0f - bias), value * bias);
}

/* ============================================================================================== */

#ifndef OCL_KERNEL
RBMSharedFunc sf_empty = { "", NULL };

RBMSharedFunc sf_binary_pdf = { "pdf", &rbm_binary_pdf };

RBMSharedFunc sf_binary_entropy = { "entropy", &rbm_binary_entropy };

RBMSharedFunc sf_binary_energy = { "energy", &rbm_basic_unit_energy };

RBMSharedFunc sf_gaussian_unit_variance_entropy = { "entropy",
  &rbm_gaussian_unit_variance_entropy };

RBMSharedFunc sf_gaussian_energy = { "energy", &rbm_gaussian_unit_energy };

RBMSharedFunc sf_gaussian_entropy = { "entropy",
  &rbm_gaussian_unit_variance_entropy };

// Keep these definitions in the same order as the RBMUnitType enum

UnitConfig rbm_config_binary = { "binary"
    , true
    , 0
    , 0
    , &rbm_binary_unit
    , &rbm_weighted_inner_sum };

UnitConfig rbm_config_rectified_linear = { "rectified_linear"
    , true
    , 0
    , 0
    , &rbm_rectified_linear_unit
    , &rbm_weighted_inner_sum };

/* No this isn't setup yet... */
UnitConfig rbm_config_simple_neurons = { "simple_neuron"
    , false
    , 4
    , 4
    , &rbm_simple_neuron_unit
    , &rbm_weighted_inner_sum };

UnitConfig rbm_config_gaussian = { "gaussian"
    , false
    , 0
    , 0
    , &rbm_gaussian_unit
    , &rbm_weighted_inner_sum };

UnitCfgMap global_rbm_all_units_config( {
{ rbm_config_binary.name, rbm_config_binary },
{ rbm_config_simple_neurons.name, rbm_config_simple_neurons },
{ rbm_config_rectified_linear.name, rbm_config_rectified_linear },
{ rbm_config_gaussian.name, rbm_config_gaussian } });

}

#endif // END of ifndef OCL_KERNEL

