#include <ailab/opencl.h>
#include <gflags/gflags.h>

DEFINE_int32(
    oclDevice, 0,
    "Use OpenCL device number. (0) for no OpenCL, (-1) for interactive prompt");
DEFINE_bool(opencl_profiling, false, "Enable profiling of OpenCL");

namespace ailab {

OpenCL::spContext setup_opencl() {
  return OpenCL::setup(FLAGS_oclDevice, FLAGS_opencl_profiling);
}

}
