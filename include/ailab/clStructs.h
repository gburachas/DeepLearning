#ifndef CLSTRUCTS_H
#define CLSTRUCTS_H

#ifndef OCL_KERNEL

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

namespace ailab{

#else
/* On the OpenCL side we need to define cl_* to enable the struct below */

typedef float cl_float;
typedef unsigned int cl_uint;
typedef unsigned char cl_uchar;

#endif

typedef struct {
  cl_float epsilon;
  cl_float epsilonDivBatch;
  cl_float decay;
  cl_float biasDecay;
  cl_float sparsityTarget;
  cl_float sparsityCost;
  cl_float weightCost;
  cl_float momentum;
  cl_uint  gibbs;
  cl_uint  batchSize;
  cl_uint  statLen;
  cl_uchar symWeights;
  cl_uint  nvis;
  cl_uint  nhid;
} RBMParams;

#ifndef OCL_KERNEL
}
#endif

#endif
