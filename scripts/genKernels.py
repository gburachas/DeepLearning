#!/usr/bin/python
import os, sys
from string import Template
import string
import re
from random import randint
from collections import namedtuple
from pprint import pprint

pathname = os.path.abspath( os.path.dirname(sys.argv[0]) )

"""
This is the source code for all of the OpenCL Kernels
It is here to facilitate extension with new unit types. OpenCL doesn't support
function pointers, which leaves us in need of an alternative.
"""

output_fname = os.path.join(pathname, "../rbm.kernels.cl")
shared_unit_cpp = os.path.join(pathname, "../common/algorithms/rbm.shared.units.cpp")
rbm_units_file = open(shared_unit_cpp)
c_inc_file = os.path.join(pathname, "../include/ailab/algorithms/rbm.kernels.source.h")

headerFiles = map(lambda x: os.path.join(pathname, x), ["../include/ailab/clStructs.h"])

# Parse definitions from shared.units.file
units_file_source = rbm_units_file.read()
raw_units_source = [ l.strip() for l in units_file_source.split("\n") ]
raw_units_file = "".join(raw_units_source)
rbm_units_file.close()

func_defs = re.findall(r"RBMSharedFunc\s+(\w+)\s+=\s+\{\s*\"(\w*)\"\s*,\s*&?(\w+)\s*\};", raw_units_file)
cfg_defs = re.findall(r"UnitConfig\s+\w+\s*=.+?\};", raw_units_file)

funcs = dict([ (fnc, (fname, fref)) for fnc,fname,fref in func_defs ])
unit_type_defs = []

for unit_cfg in cfg_defs:
      parts = re.findall(r"\w+", unit_cfg)
      
      dct = {
             "name" : parts[2]
            ,"can_sample"  : parts[3] == "true"
            ,"param_count" : parts[4]
            ,"unit_param_count": parts[5]
            ,"activation"  : parts[6]
            ,"inner"  : parts[7]
      }
      
      unit_type_defs.append( dct )


header = """
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

"""

# Generating  Kernel Signiture
activation_kernel = Template("""
/*------------------------------------------------------------------
    Generating ${name}
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel generate_${name}_unit(
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
		val += ${inner}( input[ IDX(r, i, input_pitch) ], weights[ IDX(i, unit, output_pitch) ], myState, myParams, typeParams );
	}

	if( (params.sparsityTarget > 0) && (is_visible == 0) ){
		val += (params.sparsityTarget - biasExp[unit].x) * params.sparsityCost;
	}

	output[ IDX(r, unit, output_pitch) ] = ${activation} (val, bias, myState, myParams, typeParams);
}
""")

# Sampling kernel
sampling_kernel = Template("""
/*------------------------------------------------------------------
    ${name} Sampling
    Dims: 2
    Parallel over: Units x Rows
-------------------------------------------------------------------*/
void __kernel sample_${name}_unit(
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
""")

core_kernels = """
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
"""

output_file = open(output_fname, "w")

output_file.write(header)
output_file.write(units_file_source)
output_file.write(core_kernels)

for dct in unit_type_defs:
      
      output_file.write("\n/*------------------ Begin %s Kernels --------------------------*/\n" % dct["name"] )
      output_file.write( activation_kernel.substitute(dct) )
      
      if dct["can_sample"]:
            output_file.write( sampling_kernel.substitute(dct) )
      
      output_file.write("\n/*------------------ End %s Kernels --------------------------*/\n" % dct["name"] )

output_file.close()

k_guard = "K%s" % randint(1, 10000);
f = open(c_inc_file,"w")
f.write("""#ifndef KERNELS_SOURCE_H
#define KERNELS_SOURCE_H
const char * ailab_rbm_kernels_source = R"%s(""" % k_guard)
f.close()

for fname in headerFiles:
      os.system('cat %s >> %s' % (fname, c_inc_file))

os.system('cat %s >> %s' % (output_fname, c_inc_file))

f = open(c_inc_file,"a")
f.write(")%s\";\n#endif" % k_guard)
f.close()

print("\tDone writing kernels\n")
