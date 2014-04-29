
// We only include the implementation, not the header
// because the header is intended for use by C++ on the CPU
// and is not compatible with OpenCL

typedef float decimal_t;
typedef float2 decimal2;
typedef float4 decimal4;
typedef uint uint32_t;

typedef float2 rbm_unit_state_t;
typedef float4 rbm_unit_params_t;

#define IDX(row,col,cols) ((row*cols)+col)

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


/** Functions which are not specific to a unit type **/
/*------------------------------------------------------------------
    Kernel Update biases
    Dims: 1
    Parallel over: Number of Units
-------------------------------------------------------------------*/
void __kernel update_bias(
            __global decimal_t * biases
     ,      __global decimal_t  * biasExp
     ,const __global decimal_t * data
     ,const __global decimal_t * recon
	 ,const          uchar     is_visible
     ,const          RBMParams params
     ,const          uint      pitch)
{
    uint idx;
    int r;
    const size_t unit = get_global_id(0);

    decimal_t on_data  = 0;
	decimal_t on_recon = 0;
    decimal_t bias = biases[unit];

    for(r=0; r < params.batchSize; r++)
    {
        idx = IDX(r, unit, pitch);
        on_data += data[idx];
		on_recon += recon[idx];
    }
	
	if( params.sparsityTarget > 0 && is_visible == 0 ){
		biasExp[unit] = (biasExp[unit] * params.biasDecay) + (1.0 - params.biasDecay) * (on_recon / params.batchSize);
	}
	
	bias += params.epsilonDivBatch * (on_data - on_recon);
	biases[ unit ] = bias;
}

/*------------------------------------------------------------------
    Kernel Update weights
    Dims: 2
    Parallel over: Visible x Hidden
-------------------------------------------------------------------*/
void __kernel update_weights(
            __global decimal_t * weights
     ,      __global decimal_t * velocity
     ,      __global decimal_t * weightsT
     ,      __global decimal_t * velocityT
     ,const __global decimal_t * v_data
     ,const __global decimal_t * v_recon
     ,const __global decimal_t * h_data
     ,const __global decimal_t * h_recon
	 ,const            RBMParams p)
{
    uint r;
    uint v_idx;
    uint h_idx;
    size_t v = get_global_id(0);
    size_t h = get_global_id(1);
    uint w_idx = IDX(v, h, p.nhid);
    
    decimal_t w =  weights[ w_idx ];
    decimal_t exp_sisj_data=0;
    decimal_t exp_sisj_model=0;
    decimal_t delta_ij=0;
	decimal_t vel=0;

    for(r=0; r < p.batchSize; r++)
    {
        v_idx = IDX(r, v, p.nvis);
        h_idx = IDX(r, h, p.nhid);

        exp_sisj_data += v_data[ v_idx ] * h_data[ h_idx ];
        exp_sisj_model += v_recon[ v_idx ] * h_recon[ h_idx ];
    }

	delta_ij = (exp_sisj_data - exp_sisj_model);
    delta_ij = p.epsilonDivBatch * (delta_ij - (p.decay * sign(delta_ij)) );

	if(p.momentum > 0){
	  vel = (p.momentum * velocity[w_idx]) + delta_ij;
	  delta_ij = vel;
	  velocity[ w_idx ] = vel;
	}	

	w += delta_ij;
	
    weights[ w_idx ] = w;
    
    // Everybody gets the same parameter, so
    // there is no branching cost here
    if(p.symWeights == 1){
      weightsT[ IDX(h, v, p.nvis) ] =  w;
      if(p.momentum > 0){
          velocityT[ IDX(h, v, p.nvis) ] = delta_ij;
      }
    }
	
}

/*------------------------------------------------------------------
    Entropy
    Dims: 1
    Parallel over: Visible Units
-------------------------------------------------------------------*/
void __kernel layer_entropy (__global decimal_t * results
            ,const  __global decimal_t * data
            ,const  __global decimal_t * recon
            ,const           uint        ncols
            ,const           uint        nrows )
{
      decimal_t pd = 0.0;
      decimal_t pr = 0.0;
      size_t r = 0;
      size_t c = get_global_id(0);
      for(r=0; r < nrows; r++){
          pd += data [ IDX(r, c, ncols) ];
          pr += recon[ IDX(r, c, ncols) ];
      }
      
      pd /= nrows;
      pr /= nrows;
      results[ IDX(0, c, ncols) ]  = (pd == 0.0)? 0.0 : log(pd)*pd; 
      results[ IDX(1, c, ncols) ] = (pr == 0.0)? 0.0 : log(pr)*pr;
}

/*------------------------------------------------------------------
    Free Energy of Vector: Step 1
    Dims: 1
    Parallel over: Visible Units
-------------------------------------------------------------------*/
void __kernel energy_i (__global decimal_t * energy_i
				,const  __global decimal_t * visible
				,const  __global decimal_t * visible_bias
				,const            RBMParams params)
{
	size_t ridx;
	size_t r=0;
	size_t i = get_global_id(0);
	for(r=0; r < params.batchSize; r++){
		ridx = IDX(r, i, params.nvis);
		energy_i[ridx] = visible[ridx] * visible_bias[i];	
	}	
}

/*------------------------------------------------------------------
    Free Energy of Vector: Step 2, setup partial
    Dims: 1
    Parallel over: Hidden Units
-------------------------------------------------------------------*/
void __kernel energy_j (__global decimal_t * energy_j
				,const  __global decimal_t * visible
				,const  __global decimal_t * hidden_bias
				,const  __global decimal_t * weights_vxh
				,const            RBMParams params)
{
	size_t r=0;
	size_t i;
	size_t j = get_global_id(0);
	decimal_t jbias = hidden_bias[j];
	decimal_t xj = 0;
	for(r=0; r < params.batchSize; r++){
		xj = jbias;
		for(i=0; i < params.nvis; i++){
			xj += visible[ IDX(r, i, params.nvis) ] * weights_vxh[ IDX(i, j, params.nhid) ];
		}
		energy_j[ IDX(r,j, params.nhid) ] = log(1 + exp(xj));	
	}	
}


/*------------------------------------------------------------------
    Finalize Free Energy Calculations
    Dims: 1
    Parallel over: Rows
-------------------------------------------------------------------*/
void __kernel energy_final(
		   __global decimal_t * energy
    ,const __global decimal_t * energy_i
    ,const __global decimal_t * energy_j
	,const          RBMParams   params)
{
	size_t row = get_global_id(0);
	decimal_t sumi = 0.0;
	decimal_t sumj = 0.0;
	
	for(int i=0; i < params.nvis; i++)
	{
	  sumi += energy_i[IDX(row, i, params.nvis)];
	}
	
	for(int j=0; j < params.nhid; j++)
	{
	  sumj += energy_j[IDX(row, j, params.nhid)];
	}
	
	energy[row] = - sumi - sumj;
}

/*------------------------------------------------------------------
    Make the weight decay penalty term
    Dims: 1
    Parallel over: Weights
-------------------------------------------------------------------*/
void __kernel weight_penelty_term(
			__global   decimal_t * rt
    ,const             RBMParams   params
    ,const  __global   decimal_t * weights ) {
	
	size_t i;	
	size_t j = get_global_id(0);
	decimal_t si=0.0;
	
	for(i=0; i < params.nvis; i++){
		si += pow(weights[ IDX(i, j, params.nhid) ], 2);
	}
	
	rt[j] = si;
}

/*------------------------------------------------------------------
    Normalize a portion of a layer by column
    Dims: 1
    Parallel over: Units
-------------------------------------------------------------------*/
void __kernel norm_by_column(
     __global decimal_t *   layer
    ,const    uint          pitch
    ,const    uint          batch_size)
{
    const size_t unit = get_global_id(0);
    decimal_t z=0;
    uint i,r;
    
    __global decimal_t * row;
    
    z=0.0;
    for(r=0; r < batch_size; r++)
    {
		z += layer[IDX(r,unit,pitch)];
	}
    
    z = 1.0f/z;
  
    for(r=0; r < batch_size; r++)
    {
	  layer[IDX(r,unit,pitch)] *= z;
    }
}

/*------------------------------------------------------------------
    Calc Mean Absolute Error by columns
    Dims: 1
    Parallel over: Columns
-------------------------------------------------------------------*/
void __kernel set_column_error(
     const   __global   decimal_t *       A
    ,const   __global   decimal_t *       B
    ,        __global   decimal_t *       error
    ,const              uint          pitch
    ,const              uint          batch_size)
{
      decimal_t total_error = 0.0f;
      size_t unit = get_global_id(0);
      size_t index;
      uint row;
      
      for(row=0; row < batch_size; row++)
      {
            index = IDX(row,unit,pitch);
            total_error += fabs( A[index] - B[index] );
      }
      
      error[unit] = total_error / batch_size;
}

/*------------------------------------------------------------------
    Calc Mean Absolute Error for a row
    Dims: 1
    Parallel over: Rows
-------------------------------------------------------------------*/
void __kernel set_row_error(
     const   __global   decimal_t *   A
    ,const   __global   decimal_t *   B
    ,        __global   decimal_t *   error
    ,const              uint          pitch
    ,const              uint          batch_size)
{
      decimal_t total_error = 0.0f;
      uint r = get_global_id(0);

      uint i;
      const __global decimal_t * Arow = A + (r*pitch);
      const __global decimal_t * Brow = B + (r*pitch);
      for(i=0; i < pitch; i++)
      {
            total_error += fabs( Arow[i] - Brow[i] );
      }
      error[r] = total_error / pitch;
}

/*------------------------------------------------------------------
    Naive Transpose
    Dims: 2
    Parallel over: rows x columns
-------------------------------------------------------------------*/
void __kernel naive_transpose (
      __global decimal_t * M
      ,__global decimal_t * T
      ,const uint m_rows
      ,const uint m_cols )
{
	T[ IDX(get_global_id(0), get_global_id(1), get_global_size(1)) ]
		  = M[ IDX(get_global_id(1), get_global_id(0), get_global_size(0)) ];
}

/*------------------------------------------------------------------
    Gradient reconstruction with mask
    Dims: 2
    Parallel over: rows x columns
-------------------------------------------------------------------*/
void __kernel masking_gradient_reconstruct(
	       __global decimal_t * data
	,const __global decimal_t * recon
	,      __global decimal_t * mask
	,      __global uint      * mask_sum
	,const          decimal_t   epsilon
	,const          uint        pitch)
{
	size_t row = get_global_id(0);
	size_t col = get_global_id(1);

	size_t i = IDX(row, col, pitch);
	decimal_t grad = mask[i] * (recon[i] - data[i]) * epsilon;
	uint positive_grad = isgreater(grad, 0.0);
	grad *= positive_grad;
	mask[i] = positive_grad;
	data[i] += grad;
	atomic_add(mask_sum, positive_grad);
}

/*------------------------------------------------------------------
    Gradient reconstruction
    Dims: 2
    Parallel over: rows x columns
-------------------------------------------------------------------*/
void __kernel gradient_reconstruct(
	       __global decimal_t * data
	,const __global decimal_t * recon
	,const          decimal_t   epsilon
	,const          uint        pitch)
{
	size_t i = IDX(get_global_id(0), get_global_id(1), pitch);
	data[i] += (recon[i] - data[i]) * epsilon;
}

/*------------------ Begin binary Kernels --------------------------*/

/*------------------------------------------------------------------
    Generating binary
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel generate_binary_unit(
     const __global   decimal_t *  input
    ,      __global   decimal_t *  output
    ,const __global   decimal_t *  biases
    ,const __global   decimal4  *  biasExp
    ,const __global   decimal_t *  weights
    ,      __global   decimal2  *  unitState
    ,      __global   decimal4  *  unitParams
    ,      __constant decimal_t *  typeParams
	, 	   __local    decimal_t *  row
	,const            uchar        is_visible
	,const            RBMParams    params)
{

	const unsigned int input_pitch = (is_visible == 0) ? params.nvis : params.nhid;
	const unsigned int output_pitch = (is_visible == 1) ? params.nvis : params.nhid;

	const size_t unit = get_global_id(0);      
	const size_t r = get_global_id(1);
	decimal_t bias = biases[unit];
	decimal_t val = 0.0;
	
	__global decimal2 * myState  = 0;
	__global decimal4 * myParams = 0;
	if(unitState != 0){
		myState = unitState + unit;
	}
	
	if(unitParams != 0){
		myParams = unitParams + unit;
	}
	
		// Now cache the row
	for(uint i=get_local_id(0); i < input_pitch; i += get_local_size(0)) {
		row[i] = input[ IDX(r, i, input_pitch) ];
	}
	
	for(uint i=0; i < input_pitch; i++)
	{
		val += rbm_weighted_inner_sum( input[ IDX(r, i, input_pitch) ], weights[ IDX(i, unit, output_pitch) ], myState, myParams, typeParams );
	}

	if( (params.sparsityTarget > 0) && (is_visible == 0) ){
		val += (params.sparsityTarget - biasExp[unit].x) * params.sparsityCost;
	}

	output[ IDX(r, unit, output_pitch) ] = rbm_binary_unit (val, bias, myState, myParams, typeParams);
}

/*------------------------------------------------------------------
    binary Sampling
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel sample_binary_unit(
    const  __global   decimal_t  *  input
    ,const  __global  decimal_t  *  randomNumbers
    ,      __global   decimal_t  *  output
    ,const            uint          pitch
	,const            RBMParams     params)
{
	size_t unit = get_global_id(0);
	size_t row = get_global_id(1);
	size_t i = IDX(row,unit,pitch);
	output[ i ] = 1.0 * (input[i] > randomNumbers[i]);
}

/*------------------ End binary Kernels --------------------------*/

/*------------------ Begin rectified_linear Kernels --------------------------*/

/*------------------------------------------------------------------
    Generating rectified_linear
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel generate_rectified_linear_unit(
     const __global   decimal_t *  input
    ,      __global   decimal_t *  output
    ,const __global   decimal_t *  biases
    ,const __global   decimal4  *  biasExp
    ,const __global   decimal_t *  weights
    ,      __global   decimal2  *  unitState
    ,      __global   decimal4  *  unitParams
    ,      __constant decimal_t *  typeParams
	, 	   __local    decimal_t *  row
	,const            uchar        is_visible
	,const            RBMParams    params)
{

	const unsigned int input_pitch = (is_visible == 0) ? params.nvis : params.nhid;
	const unsigned int output_pitch = (is_visible == 1) ? params.nvis : params.nhid;

	const size_t unit = get_global_id(0);      
	const size_t r = get_global_id(1);
	decimal_t bias = biases[unit];
	decimal_t val = 0.0;
	
	__global decimal2 * myState  = 0;
	__global decimal4 * myParams = 0;
	if(unitState != 0){
		myState = unitState + unit;
	}
	
	if(unitParams != 0){
		myParams = unitParams + unit;
	}
	
		// Now cache the row
	for(uint i=get_local_id(0); i < input_pitch; i += get_local_size(0)) {
		row[i] = input[ IDX(r, i, input_pitch) ];
	}
	
	for(uint i=0; i < input_pitch; i++)
	{
		val += rbm_weighted_inner_sum( input[ IDX(r, i, input_pitch) ], weights[ IDX(i, unit, output_pitch) ], myState, myParams, typeParams );
	}

	if( (params.sparsityTarget > 0) && (is_visible == 0) ){
		val += (params.sparsityTarget - biasExp[unit].x) * params.sparsityCost;
	}

	output[ IDX(r, unit, output_pitch) ] = rbm_rectified_linear_unit (val, bias, myState, myParams, typeParams);
}

/*------------------------------------------------------------------
    rectified_linear Sampling
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel sample_rectified_linear_unit(
    const  __global   decimal_t  *  input
    ,const  __global  decimal_t  *  randomNumbers
    ,      __global   decimal_t  *  output
    ,const            uint          pitch
	,const            RBMParams     params)
{
	size_t unit = get_global_id(0);
	size_t row = get_global_id(1);
	size_t i = IDX(row,unit,pitch);
	output[ i ] = 1.0 * (input[i] > randomNumbers[i]);
}

/*------------------ End rectified_linear Kernels --------------------------*/

/*------------------ Begin simple_neuron Kernels --------------------------*/

/*------------------------------------------------------------------
    Generating simple_neuron
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel generate_simple_neuron_unit(
     const __global   decimal_t *  input
    ,      __global   decimal_t *  output
    ,const __global   decimal_t *  biases
    ,const __global   decimal4  *  biasExp
    ,const __global   decimal_t *  weights
    ,      __global   decimal2  *  unitState
    ,      __global   decimal4  *  unitParams
    ,      __constant decimal_t *  typeParams
	, 	   __local    decimal_t *  row
	,const            uchar        is_visible
	,const            RBMParams    params)
{

	const unsigned int input_pitch = (is_visible == 0) ? params.nvis : params.nhid;
	const unsigned int output_pitch = (is_visible == 1) ? params.nvis : params.nhid;

	const size_t unit = get_global_id(0);      
	const size_t r = get_global_id(1);
	decimal_t bias = biases[unit];
	decimal_t val = 0.0;
	
	__global decimal2 * myState  = 0;
	__global decimal4 * myParams = 0;
	if(unitState != 0){
		myState = unitState + unit;
	}
	
	if(unitParams != 0){
		myParams = unitParams + unit;
	}
	
		// Now cache the row
	for(uint i=get_local_id(0); i < input_pitch; i += get_local_size(0)) {
		row[i] = input[ IDX(r, i, input_pitch) ];
	}
	
	for(uint i=0; i < input_pitch; i++)
	{
		val += rbm_weighted_inner_sum( input[ IDX(r, i, input_pitch) ], weights[ IDX(i, unit, output_pitch) ], myState, myParams, typeParams );
	}

	if( (params.sparsityTarget > 0) && (is_visible == 0) ){
		val += (params.sparsityTarget - biasExp[unit].x) * params.sparsityCost;
	}

	output[ IDX(r, unit, output_pitch) ] = rbm_simple_neuron_unit (val, bias, myState, myParams, typeParams);
}

/*------------------ End simple_neuron Kernels --------------------------*/

/*------------------ Begin gaussian Kernels --------------------------*/

/*------------------------------------------------------------------
    Generating gaussian
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel generate_gaussian_unit(
     const __global   decimal_t *  input
    ,      __global   decimal_t *  output
    ,const __global   decimal_t *  biases
    ,const __global   decimal4  *  biasExp
    ,const __global   decimal_t *  weights
    ,      __global   decimal2  *  unitState
    ,      __global   decimal4  *  unitParams
    ,      __constant decimal_t *  typeParams
	, 	   __local    decimal_t *  row
	,const            uchar        is_visible
	,const            RBMParams    params)
{

	const unsigned int input_pitch = (is_visible == 0) ? params.nvis : params.nhid;
	const unsigned int output_pitch = (is_visible == 1) ? params.nvis : params.nhid;

	const size_t unit = get_global_id(0);      
	const size_t r = get_global_id(1);
	decimal_t bias = biases[unit];
	decimal_t val = 0.0;
	
	__global decimal2 * myState  = 0;
	__global decimal4 * myParams = 0;
	if(unitState != 0){
		myState = unitState + unit;
	}
	
	if(unitParams != 0){
		myParams = unitParams + unit;
	}
	
		// Now cache the row
	for(uint i=get_local_id(0); i < input_pitch; i += get_local_size(0)) {
		row[i] = input[ IDX(r, i, input_pitch) ];
	}
	
	for(uint i=0; i < input_pitch; i++)
	{
		val += rbm_weighted_inner_sum( input[ IDX(r, i, input_pitch) ], weights[ IDX(i, unit, output_pitch) ], myState, myParams, typeParams );
	}

	if( (params.sparsityTarget > 0) && (is_visible == 0) ){
		val += (params.sparsityTarget - biasExp[unit].x) * params.sparsityCost;
	}

	output[ IDX(r, unit, output_pitch) ] = rbm_gaussian_unit (val, bias, myState, myParams, typeParams);
}

/*------------------ End gaussian Kernels --------------------------*/
