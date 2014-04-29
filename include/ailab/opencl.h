#ifndef OPENCL_H
#define OPENCL_H

#include <assert.h>
#include <string>
#include <string.h>
#include <cstdio>
#include <fstream>
#include <iostream>
#include <sstream>
#include <memory>
#include <map>
#include <vector>
#include <list>
#include <set>

#ifdef __APPLE__
#include <OpenCL/opencl.h>
#else
#include <CL/cl.h>
#endif

#include <ailab/clStructs.h>

namespace ailab {

class OpenCL {

 public:
  class KernelLocalMemory;

  class Queue;
  class Device;
  class Context;
  class Platform;
  class Worker;
  class Memory;
  class Kernel;
  class Error;

  typedef std::shared_ptr<Queue> spQueue;
  typedef std::shared_ptr<Device> spDevice;
  typedef std::shared_ptr<Context> spContext;
  typedef std::shared_ptr<Platform> spPlatform;
  typedef std::shared_ptr<Memory> spMemory;
  typedef std::shared_ptr<Kernel> spKernel;

  typedef std::pair<std::string, spKernel> KernelMapPair;

  typedef std::map<std::string, spKernel> KernelMap;

  static bool isset(int flag, int mask) {
    return (mask & flag) == flag;
  }

  typedef struct {
    cl_ulong enqueued;
    cl_ulong submitted;
    cl_ulong started;
    cl_ulong ended;
  } CL_PROFILING_TIMES;

  class KernelLocalMemory {
   public:
    KernelLocalMemory(size_t size) {
      this->size = size;
    }
    size_t size;
  };

  class Platform {

    cl_platform_id _cl;

    template<typename T>
    T * _get(cl_platform_info param) {
      size_t size;
      clGetPlatformInfo(this->_cl, param, 0, NULL, &size);
      void * buffer = malloc(size);
      clGetPlatformInfo(this->_cl, param, size, buffer, NULL);
      return (T *) buffer;
    }

    void _clear() {
      this->devices.clear();

      if (this->profile != NULL) {
        free(this->profile);
        this->profile = NULL;
      }
      if (this->version != NULL) {
        free(this->version);
        this->version = NULL;
      }
      if (this->name != NULL) {
        free(this->name);
        this->name = NULL;
      }
      if (this->vender != NULL) {
        free(this->vender);
        this->vender = NULL;
      }
      if (this->extensions != NULL) {
        free(this->extensions);
        this->extensions = NULL;
      }
    }

   public:

    Platform() {
      this->_cl = NULL;
      this->profile = NULL;
      this->version = NULL;
      this->name = NULL;
      this->vender = NULL;
      this->extensions = NULL;
    }

    ~Platform() {
      this->_clear();
    }

    operator cl_platform_id() {
      return this->_cl;
    }

    void init(cl_platform_id id) {
      this->_cl = id;

      this->_clear();
      this->profile = this->_get<char>(CL_PLATFORM_PROFILE);
      this->version = this->_get<char>(CL_PLATFORM_VERSION);
      this->name = this->_get<char>(CL_PLATFORM_NAME);
      this->vender = this->_get<char>(CL_PLATFORM_VENDOR);
      this->extensions = this->_get<char>(CL_PLATFORM_EXTENSIONS);

      cl_uint n_devices;
      clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, 0, NULL, &n_devices);
      this->devices.resize(n_devices);
      clGetDeviceIDs(id, CL_DEVICE_TYPE_ALL, n_devices, &this->devices[0],
      NULL);
    }

    std::vector<cl_device_id> devices;

    char * profile;
    char * version;
    char * name;
    char * vender;
    char * extensions;

  };

  class Worker {

   protected:

    unsigned int ncalls;

   public:

    cl_context context;
    cl_command_queue queue;

    Worker(Device * device)
        : device(device),
          context(NULL),
          ncalls(0) {
    }

    ~Worker() {
    }

    void init(cl_context context, cl_command_queue queue) {
      this->context = context;
      this->queue = queue;
    }

    void dump_profiling_info(std::ostream * s) {
      *s << "Enqueued,dSubmitted,dStarted,dEnded" << std::endl;

      CL_PROFILING_TIMES t;
      for (std::list<CL_PROFILING_TIMES>::iterator iter =
          this->perf_times.begin(); iter != this->perf_times.end(); iter++) {
        t = *iter;
        *s << t.enqueued << "," << t.submitted - t.enqueued << ","
           << t.started - t.submitted << "," << t.ended - t.started
           << std::endl;
      }

      *s << std::flush;
    }

    Device *device;
    std::list<CL_PROFILING_TIMES> perf_times;
  };

  class Memory : public Worker {
   protected:
    bool _initialized;
    cl_mem _read_only_mem;
    cl_mem cl_dev_mem;

    size_t read_count;
    size_t write_count;

   public:

    Memory(Device *device)
        : Worker(device),
          read_count(0),
          write_count(0),
          _initialized(false) {
      this->touched_by_device = false;
      this->touched_by_host = false;
      this->host = NULL;
      this->size = 0;
      this->cl_dev_mem = NULL;
    }

    void * getHost() {
      this->read();
      this->touched_by_host = true;
      return this->host;
    }

    const void * getHostConst() {
      // Do not set the touched_by_host flag
      // This will allow us to read data and use it
      // without constantly syncing to and from the device
      this->read();
      return this->host;
    }

    cl_mem getCL() {
      if (this->cl_dev_mem != NULL) {
        this->write();
        this->touched_by_device = true;
        return this->_read_only_mem;
      } else
        return NULL;
    }

    const cl_mem getCLConst() {
      if (this->cl_dev_mem != NULL) {
        this->write();
        return this->_read_only_mem;
      } else
        return NULL;
    }

    void init(Memory& parent) {
      this->init(parent, 0, this->size);
    }

    void init(Memory& parent, size_t offset, size_t size) {
      if (!this->_initialized) {
        cl_int stat;

        this->_initialized = true;
        this->context = parent.context;
        this->queue = parent.queue;
        this->owns_host = false;
        cl_buffer_region region = { offset, size };
        this->size = size;
        this->host = ((char*) parent.host) + offset;
        this->cl_dev_mem = clCreateSubBuffer(parent.cl_dev_mem, parent.flags,
        CL_BUFFER_CREATE_TYPE_REGION,
                                             &region, &stat);
        if (stat != CL_SUCCESS)
          exit(stat);
        Worker::init(parent.context, parent.queue);
      }
    }

    void init(cl_context context, cl_command_queue queue, size_t size,
              cl_mem_flags flags = CL_MEM_READ_WRITE, void * host_mem = NULL) {
      if (!this->_initialized) {
        this->_initialized = true;
        this->context = context;
        this->queue = queue;
        cl_int stat = CL_SUCCESS;
        this->size = size;

        if (host_mem == NULL) {
          this->cl_dev_mem = clCreateBuffer(context,
                                            flags | CL_MEM_ALLOC_HOST_PTR, size,
                                            NULL,
                                            &stat);
          assert(stat == CL_SUCCESS);
          this->host = clEnqueueMapBuffer(queue, this->cl_dev_mem, CL_TRUE,
          CL_MAP_WRITE | CL_MAP_READ,
                                          0, size, 0, NULL, NULL, &stat);
          assert(stat == CL_SUCCESS);
          this->owns_host = true;

        } else {

          this->cl_dev_mem = clCreateBuffer(context, flags, size, NULL, &stat);
          assert(stat == CL_SUCCESS);
          this->host = host_mem;
          this->owns_host = false;
        }

        if (stat != CL_SUCCESS)
          exit(stat);

        cl_buffer_region subregion = { 0, size };

        this->_read_only_mem = clCreateSubBuffer(this->cl_dev_mem,
        CL_MEM_READ_ONLY,
                                                 CL_BUFFER_CREATE_TYPE_REGION,
                                                 &subregion, &stat);
        if (stat != CL_SUCCESS)
          exit(stat);
        Worker::init(context, queue);
        this->load();

      }
    }

    operator cl_mem() {
      return this->cl_dev_mem;
    }

    void operator=(Memory& other) {
      this->touched_by_device = true;
      clEnqueueCopyBuffer(this->queue, other.getCLConst(), this->cl_dev_mem, 0,
                          0, this->size, 0, NULL, NULL);
    }

    void operator=(void* other) {
      this->touched_by_host = true;
      memmove(this->host, other, this->size);
    }

    ~Memory() {
      clReleaseMemObject(this->cl_dev_mem);
      this->device->memory_available += this->size;
    }

    void load() {
      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_SIZE, sizeof(size_t),
                         &(this->size), NULL);
      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_OFFSET, sizeof(size_t),
                         &this->offset, NULL);
      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_FLAGS, sizeof(cl_mem_flags),
                         &this->flags, NULL);

      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_MAP_COUNT, sizeof(cl_uint),
                         &this->map_cnt, NULL);
      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_REFERENCE_COUNT,
                         sizeof(cl_uint), &this->mem_ref_count, NULL);

      clGetMemObjectInfo(this->cl_dev_mem, CL_MEM_HOST_PTR, sizeof(void*),
                         &this->cl_host_ptr, NULL);

      this->is_readable_by_kernel = isset(CL_MEM_READ_ONLY, this->flags)
          || isset(CL_MEM_READ_WRITE, this->flags);
      this->is_writable_by_kernel = isset(CL_MEM_WRITE_ONLY, this->flags)
          || isset(CL_MEM_READ_WRITE, this->flags);
    }

    Memory slice(size_t offset, size_t length_in_bytes) {
      Memory sub(this->device);
      sub.init(*this, offset, length_in_bytes);
      return sub;
    }

    void read() {
      this->read(this->host, this->size);
    }

    void read(void * dest, size_t size) {
      if (this->touched_by_device) {
        cl_int status = clEnqueueReadBuffer(this->queue,
                                            this->cl_dev_mem, CL_TRUE,
                                            0, size, dest, 0, NULL, NULL);
        OpenCL::stat(status, "Could not read buffer: ");
        this->read_count++;
        this->touched_by_device = false;
      }
    }

    size_t getReadCount(){
      return this->read_count;
    }

    void write() {
      this->write(this->host, this->size);
    }

    void write(void *buf) {
      this->write(buf, this->size);
    }

    void write(void *buf, size_t size) {
      if (this->touched_by_host) {
        cl_int status = CL_SUCCESS;
        status = clEnqueueWriteBuffer(this->queue, this->cl_dev_mem, CL_TRUE, 0,
                                      size, buf, 0, NULL, NULL);

        OpenCL::stat(status, "Could not write buffer: ");

        this->write_count++;
        if (buf != this->host)
          memmove(this->host, buf, size);

        this->touched_by_host = false;
      }
    }

    size_t getWriteCount(){
      return this->write_count;
    }

    void set(char v) {
      memset(this->host, v, this->size);
    }

    size_t offset;
    size_t size;
    void * host;
    cl_mem_flags flags;
    void * cl_host_ptr;
    cl_uint map_cnt;
    cl_uint mem_ref_count;

    bool touched_by_host;
    bool touched_by_device;
    bool owns_host;
    bool is_readable_by_kernel;
    bool is_writable_by_kernel;
  };

  class Kernel : public Worker {
   protected:

    template<typename A>
    inline void _cl_setArg(A i) {
      clSetKernelArg(this->_kernel, this->arg_idx++, sizeof(A), (void *) &i);
    }

    template<typename First, typename ... Rest>
    void _setArgs(First start, Rest ... rest) {
      _cl_setArg<First>(start);
      this->_setArgs(rest...);
    }

    void _setArgs() {
    }

    cl_kernel _kernel;
    cl_uint work_dim;
    size_t global_size[3];
    size_t local_size[3];
    size_t global_offset[3];

    size_t arg_idx;

   public:

    Kernel(Device * device)
        : Worker(device) {
      this->name = NULL;
      this->attributes = NULL;
      this->n_args = 0;
      this->work_dim = 0;
      this->global_offset[0] = 0;
      this->global_offset[1] = 0;
      this->global_offset[2] = 0;
    }

    operator cl_kernel() {
      return this->_kernel;
    }

    void init(cl_context context, cl_command_queue queue, cl_kernel kernel) {
      size_t size = 0;
      this->_kernel = kernel;
      this->queue = queue;
      this->local_mem_req = 0;
      clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, 0, NULL, &size);
      this->name = (char *) malloc(size);
      clGetKernelInfo(kernel, CL_KERNEL_FUNCTION_NAME, size, this->name, NULL);

      clGetKernelInfo(kernel, CL_KERNEL_NUM_ARGS, sizeof(cl_uint),
                      &this->n_args, NULL);

#ifdef CL_KERNEL_ATTRIBUTES
      clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, 0, NULL, &size);
      this->attributes = (char *) malloc(size);
      clGetKernelInfo(kernel, CL_KERNEL_ATTRIBUTES, size, this->attributes, NULL);
#else
      this->attributes = NULL;
#endif
      clGetKernelInfo(kernel, CL_KERNEL_CONTEXT, sizeof(cl_context),
                      &this->context, NULL);
      clGetKernelInfo(kernel, CL_KERNEL_PROGRAM, sizeof(cl_program),
                      &this->program, NULL);

      clGetKernelWorkGroupInfo(kernel, (cl_device_id) this->device,
      CL_KERNEL_LOCAL_MEM_SIZE,
                               sizeof(cl_ulong), &this->local_mem_req, NULL);
      clGetKernelWorkGroupInfo(kernel, (cl_device_id) this->device,
      CL_KERNEL_PRIVATE_MEM_SIZE,
                               sizeof(cl_ulong), &this->local_mem_req, NULL);

      if (this->local_mem_req > 0)
        this->max_workgroup_size = std::min(
            OpenCL::getWGMultiple(this),
            this->device->local_mem_size / this->local_mem_req);
      else
        this->max_workgroup_size = OpenCL::getWGMultiple(this);

      Worker::init(context, queue);
    }

    ~Kernel() {
      free(this->name);
#ifdef CL_KERNEL_ATTRIBUTES
      free(this->attributes);
#endif
    }

    void set_as_task() {
      this->work_dim = 0;
    }
    void set_offset(size_t offset1 = 0, size_t offset2 = 0,
                    size_t offset3 = 0) {
      this->global_offset[0] = offset1;
      this->global_offset[1] = offset2;
      this->global_offset[2] = offset3;
    }
    void set_size(size_t global1, size_t global2, size_t global3, size_t local1,
                  size_t local2 = 0, size_t local3 = 0) {
      this->set_size(global1, global2, global3);

      this->local_size[0] = local1;
      this->local_size[1] = local2;
      this->local_size[2] = local3;

    }

    void set_size(size_t global1, size_t global2 = 0, size_t global3 = 0) {
      size_t n_dims = 0;
      this->local_size[0] = 0;
      this->local_size[1] = 0;
      this->local_size[2] = 0;

      this->global_size[n_dims++] = global1;

      if (global2 > 0)
        this->global_size[n_dims++] = global2;

      if (global3 > 0)
        this->global_size[n_dims++] = global3;

      this->work_dim = n_dims;
    }

    void run() {
      std::stringstream s;
      cl_int err;
      if (this->arg_idx != this->n_args) {
        std::cout << "Could not run " << this->name
                  << ", wrong number of arguments " << this->arg_idx
                  << " out of " << this->n_args << std::endl;
        exit(CL_INVALID_KERNEL_ARGS);
      }

      if (this->work_dim > 0)
        err = clEnqueueNDRangeKernel(
            this->queue, this->_kernel, this->work_dim, this->global_offset,
            this->global_size,
            (this->local_size[0] == 0) ? NULL : this->local_size, 0, NULL,
            NULL);
      else
        err = clEnqueueTask(this->queue, this->_kernel, 0, NULL, NULL);

      if (err != CL_SUCCESS) {
        s << "Error enqueueing " << this->name << " ";
        OpenCL::stat(err, s.str().c_str());
        exit(err);
      }
    }

    template<typename ... Args>
    void run(Args ...args) {
      this->setArgs(args...);
      this->run();
    }

    template<typename First, typename ... Rest>
    void setArgs(First start, Rest ... rest) {
      this->arg_idx = 0;
      this->_cl_setArg(start);
      this->_setArgs(rest...);
    }

    cl_kernel get() {
      return this->_kernel;
    }

    // Public Attributes
    char * name;
    char * attributes;
    size_t n_args;
    cl_ulong local_mem_req;
    cl_ulong private_mem_req;
    size_t max_workgroup_size;
    cl_context context;
    cl_program program;
  };

  class Error {
   public:

    Error(cl_int num, const char * desc, const char * msg)
        : number(num),
          description(desc),
          message(msg) {
    }

    std::string description;
    std::string message;
    cl_int number;

  };

  class Device {
    friend class Memory;
    cl_device_id _cl;

    template<typename T>
    T * _get(cl_device_info param) {

      size_t size;
      clGetDeviceInfo(this->_cl, param, 0, NULL, &size);
      void * buf = malloc(size);
      clGetDeviceInfo(this->_cl, param, size, buf, NULL);
      return (T *) buf;
    }

   public:

    void init(cl_device_id id) {
      this->_cl = id;
      this->load();
    }

    operator cl_device_id() {
      return this->_cl;
    }

    spMemory alloc(cl_context context, cl_command_queue queue, size_t size,
                   cl_mem_flags flags = CL_MEM_READ_WRITE,
                   void* host_mem = NULL) {

      spMemory r = spMemory(new Memory(this));
      r->init(context, queue, size, flags, host_mem);
      this->memory_available -= size;
      return r;
    }

    ~Device() {

      if (this->profile != NULL)
        free(this->profile);

      if (this->vendor != NULL)
        free(this->vendor);

      if (this->cl_driver_version != NULL)
        free(this->cl_driver_version);

      if (this->version != NULL)
        free(this->version);

      if (this->extensions != NULL)
        free(this->extensions);

      if (this->name != NULL)
        free(this->name);
    }

    void load() {

      int i = 1;

      clGetDeviceInfo(this->_cl, CL_DEVICE_TYPE, (size_t) sizeof(this->type),
                      (void *) &(this->type), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_VENDOR_ID,
                      (size_t) sizeof(this->vendor_id),
                      (void *) &(this->vendor_id), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_COMPUTE_UNITS,
                      (size_t) sizeof(this->max_compute_units),
                      (void *) &(this->max_compute_units), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_WORK_ITEM_DIMENSIONS,
                      (size_t) sizeof(this->max_work_item_dimensions),
                      (void *) &(this->max_work_item_dimensions), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_WORK_GROUP_SIZE,
                      (size_t) sizeof(this->max_work_group_size),
                      (void *) &(this->max_work_group_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_WORK_ITEM_SIZES,
                      (size_t) sizeof(this->max_work_item_sizes),
                      (void *) &(this->max_work_item_sizes), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_CHAR,
                      (size_t) sizeof(this->preferred_vector_width_char),
                      (void *) &(this->preferred_vector_width_char), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_SHORT,
                      (size_t) sizeof(this->preferred_vector_width_short),
                      (void *) &(this->preferred_vector_width_short), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_INT,
                      (size_t) sizeof(this->preferred_vector_width_int),
                      (void *) &(this->preferred_vector_width_int), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_LONG,
                      (size_t) sizeof(this->preferred_vector_width_long),
                      (void *) &(this->preferred_vector_width_long), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_FLOAT,
                      (size_t) sizeof(this->preferred_vector_width_float),
                      (void *) &(this->preferred_vector_width_float), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PREFERRED_VECTOR_WIDTH_DOUBLE,
                      (size_t) sizeof(this->preferred_vector_width_double),
                      (void *) &(this->preferred_vector_width_double), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_CLOCK_FREQUENCY,
                      (size_t) sizeof(this->max_clock_frequency),
                      (void *) &(this->max_clock_frequency), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_ADDRESS_BITS,
                      (size_t) sizeof(this->address_bits),
                      (void *) &(this->address_bits), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_READ_IMAGE_ARGS,
                      (size_t) sizeof(this->max_read_image_args),
                      (void *) &(this->max_read_image_args), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_WRITE_IMAGE_ARGS,
                      (size_t) sizeof(this->max_write_image_args),
                      (void *) &(this->max_write_image_args), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_MEM_ALLOC_SIZE,
                      (size_t) sizeof(this->max_mem_alloc_size),
                      (void *) &(this->max_mem_alloc_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE2D_MAX_WIDTH,
                      (size_t) sizeof(this->image2d_max_width),
                      (void *) &(this->image2d_max_width), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE2D_MAX_HEIGHT,
                      (size_t) sizeof(this->image2d_max_height),
                      (void *) &(this->image2d_max_height), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE3D_MAX_WIDTH,
                      (size_t) sizeof(this->image3d_max_width),
                      (void *) &(this->image3d_max_width), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE3D_MAX_HEIGHT,
                      (size_t) sizeof(this->image3d_max_height),
                      (void *) &(this->image3d_max_height), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE3D_MAX_DEPTH,
                      (size_t) sizeof(this->image3d_max_depth),
                      (void *) &(this->image3d_max_depth), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_IMAGE_SUPPORT,
                      (size_t) sizeof(this->image_support),
                      (void *) &(this->image_support), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_PARAMETER_SIZE,
                      (size_t) sizeof(this->max_parameter_size),
                      (void *) &(this->max_parameter_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_SAMPLERS,
                      (size_t) sizeof(this->max_samplers),
                      (void *) &(this->max_samplers), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MEM_BASE_ADDR_ALIGN,
                      (size_t) sizeof(this->mem_base_addr_align),
                      (void *) &(this->mem_base_addr_align), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MIN_DATA_TYPE_ALIGN_SIZE,
                      (size_t) sizeof(this->min_data_type_align_size),
                      (void *) &(this->min_data_type_align_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_SINGLE_FP_CONFIG,
                      (size_t) sizeof(this->single_fp_config),
                      (void *) &(this->single_fp_config), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_GLOBAL_MEM_CACHE_TYPE,
                      (size_t) sizeof(this->global_mem_cache_type),
                      (void *) &(this->global_mem_cache_type), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_GLOBAL_MEM_CACHELINE_SIZE,
                      (size_t) sizeof(this->global_mem_cacheline_size),
                      (void *) &(this->global_mem_cacheline_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_GLOBAL_MEM_CACHE_SIZE,
                      (size_t) sizeof(this->global_mem_cache_size),
                      (void *) &(this->global_mem_cache_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_GLOBAL_MEM_SIZE,
                      (size_t) sizeof(this->global_mem_size),
                      (void *) &(this->global_mem_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_CONSTANT_BUFFER_SIZE,
                      (size_t) sizeof(this->max_constant_buffer_size),
                      (void *) &(this->max_constant_buffer_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_MAX_CONSTANT_ARGS,
                      (size_t) sizeof(this->max_constant_args),
                      (void *) &(this->max_constant_args), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_LOCAL_MEM_TYPE,
                      (size_t) sizeof(this->local_mem_type),
                      (void *) &(this->local_mem_type), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_LOCAL_MEM_SIZE,
                      (size_t) sizeof(this->local_mem_size),
                      (void *) &(this->local_mem_size), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_ERROR_CORRECTION_SUPPORT,
                      (size_t) sizeof(this->error_correction_support),
                      (void *) &(this->error_correction_support), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PROFILING_TIMER_RESOLUTION,
                      (size_t) sizeof(this->profiling_timer_resolution),
                      (void *) &(this->profiling_timer_resolution), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_ENDIAN_LITTLE,
                      (size_t) sizeof(this->endian_little),
                      (void *) &(this->endian_little), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_AVAILABLE,
                      (size_t) sizeof(this->available),
                      (void *) &(this->available), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_COMPILER_AVAILABLE,
                      (size_t) sizeof(this->compiler_available),
                      (void *) &(this->compiler_available), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_EXECUTION_CAPABILITIES,
                      (size_t) sizeof(this->execution_capabilities),
                      (void *) &(this->execution_capabilities), NULL);
      clGetDeviceInfo(this->_cl, CL_DEVICE_PLATFORM,
                      (size_t) sizeof(this->platform),
                      (void *) &(this->platform), NULL);

      this->profile = this->_get<char>(CL_DEVICE_PROFILE);
      this->vendor = this->_get<char>(CL_DEVICE_VENDOR);
      this->cl_driver_version = this->_get<char>(CL_DRIVER_VERSION);
      this->version = this->_get<char>(CL_DEVICE_VERSION);
      this->extensions = this->_get<char>(CL_DEVICE_EXTENSIONS);
      this->name = this->_get<char>(CL_DEVICE_NAME);

      this->endian_mismatch = ((*(unsigned char *) &i) != this->endian_little);

      this->memory_available = this->global_mem_size;
    }

    cl_device_id get() {
      return this->_cl;
    }

    cl_uint address_bits;
    cl_bool available;
    cl_bool compiler_available;
    cl_bool endian_little;
    cl_bool error_correction_support;
    cl_device_exec_capabilities execution_capabilities;
    cl_ulong global_mem_cache_size;
    cl_device_mem_cache_type global_mem_cache_type;
    cl_uint global_mem_cacheline_size;
    cl_ulong global_mem_size;
    cl_bool image_support;
    size_t image2d_max_height;
    size_t image2d_max_width;
    size_t image3d_max_depth;
    size_t image3d_max_height;
    size_t image3d_max_width;
    cl_ulong local_mem_size;
    cl_device_local_mem_type local_mem_type;
    cl_uint max_clock_frequency;
    cl_uint max_compute_units;
    cl_uint max_constant_args;
    cl_ulong max_constant_buffer_size;
    cl_ulong max_mem_alloc_size;
    size_t max_parameter_size;
    cl_uint max_read_image_args;
    cl_uint max_samplers;
    size_t max_work_group_size;
    cl_uint max_work_item_dimensions;
    size_t max_work_item_sizes[3];
    cl_uint max_write_image_args;
    cl_uint mem_base_addr_align;
    cl_uint min_data_type_align_size;
    cl_platform_id platform;
    cl_uint preferred_vector_width_char;
    cl_uint preferred_vector_width_double;
    cl_uint preferred_vector_width_float;
    cl_uint preferred_vector_width_int;
    cl_uint preferred_vector_width_long;
    cl_uint preferred_vector_width_short;
    size_t profiling_timer_resolution;
    cl_command_queue_properties queue_properties;
    cl_device_fp_config single_fp_config;
    cl_device_type type;
    cl_uint vendor_id;

    char * extensions;
    char * name;
    char * profile;
    char * vendor;
    char * version;
    char * cl_driver_version;

    bool endian_mismatch;
    size_t memory_available;

  };

  class Queue {
    cl_command_queue _cl;
   public:

    operator cl_command_queue() {
      return this->_cl;
    }

    virtual void init(cl_context context, cl_device_id device,
                      cl_command_queue_properties properties) {
      cl_int stat;

      this->_cl = clCreateCommandQueue(context, device, properties , &stat);

      if (stat != CL_SUCCESS)
        exit(stat);

      this->load();
    }

    void load() {

      clGetCommandQueueInfo(this->_cl, CL_QUEUE_CONTEXT, sizeof(cl_context),
                            &this->context, NULL);
      clGetCommandQueueInfo(this->_cl, CL_QUEUE_DEVICE, sizeof(cl_device_id),
                            &this->device, NULL);
      clGetCommandQueueInfo(this->_cl, CL_QUEUE_REFERENCE_COUNT,
                            sizeof(cl_uint), &this->references, NULL);

      cl_command_queue_properties props;
      clGetCommandQueueInfo(this->_cl, CL_QUEUE_PROPERTIES,
                            sizeof(cl_command_queue_properties), &props, NULL);

      this->out_of_order_exec = (props & CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE)
          == CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE;
      this->profiling = (props & CL_QUEUE_PROFILING_ENABLE)
          == CL_QUEUE_PROFILING_ENABLE;
    }

    void set_fence() {
      clEnqueueBarrier(this->_cl);
    }

    cl_command_queue get() {
      return this->_cl;
    }

    cl_context context;
    cl_device_id device;
    cl_uint references;
    bool out_of_order_exec;
    bool profiling;
  };

  class Context {
   protected:
    cl_context _cl;
    cl_program _program;

    template<typename T>
    T * _get(cl_context_info param) {

      size_t size;
      clGetContextInfo(this->_cl, param, 0, NULL, &size);
      void * buf = malloc(size);
      clGetContextInfo(this->_cl, param, size, buf, NULL);
      return (T *) buf;
    }

    KernelMap kernels;
    bool profiling_is_enabled;

   public:

    operator cl_context() {
      return this->_cl;
    }

    void init(cl_platform_id platform, cl_device_id device,
              cl_command_queue_properties properties) {
      cl_int stat = CL_SUCCESS;
      cl_context_properties p[] = { CL_CONTEXT_PLATFORM,
          (cl_context_properties) platform, 0 };
      this->_cl = clCreateContext(p, 1, &device, NULL, NULL, &stat);

      if (stat != CL_SUCCESS) {
        OpenCL::stat(stat, "Check OpenCL error: ");
        exit(stat);
      }

      this->load();

      this->device.init(device);
      this->queue.init(this->_cl, device, properties);
    }

    void load() {
      size_t dev_size;
      clGetContextInfo(this->_cl, CL_CONTEXT_DEVICES, 0, NULL, &dev_size);
      this->devices.resize(dev_size / sizeof(cl_device_id), 0);
      clGetContextInfo(this->_cl, CL_CONTEXT_DEVICES, dev_size,
                       &this->devices[0], NULL);
      clGetContextInfo(this->_cl, CL_CONTEXT_REFERENCE_COUNT, sizeof(cl_uint),
                       &this->reference_count, NULL);
    }

    spMemory alloc(size_t size, cl_mem_flags flags = CL_MEM_READ_WRITE,
                   void * host_mem = NULL) {
      return this->device.alloc(this->_cl, this->queue.get(), size, flags,
                                host_mem);
    }

    spKernel getKernel(const char * name, const char * format) {
      char buf[ strlen(name) + strlen(format) ];
      memset(buf, 0, sizeof(buf));
      sprintf(buf, format, name);
      return this->getKernel( std::string(buf, strlen(buf)) );
    }

    spKernel getKernel(const std::string name) {
         auto nIter = this->kernels.find(name);
          if (nIter == this->kernels.end())
            return NULL;
          else
            return nIter->second;
        }

    size_t loadKernels(const char * source, const std::string build_options) {
      size_t n_start_kernels = kernels.size();
      size_t size = strlen(source);
      // Build the code
      cl_int program_stat = CL_SUCCESS;
      cl_program program = clCreateProgramWithSource(this->_cl, 1, &source,
                                                     &size, &program_stat);

      OpenCL::stat(program_stat, "Error creating program: ");

      std::string buildOptions(
          "-D OCL_KERNEL ");
      buildOptions.append(build_options);

      cl_int build_stat = clBuildProgram(program, 0, 0, buildOptions.c_str(), 0,
                                         0);

      if (build_stat != CL_SUCCESS) {
        clGetProgramBuildInfo(program, this->device.get(), CL_PROGRAM_BUILD_LOG,
                              0, NULL, &size);
        char * log = new char[size];
        clGetProgramBuildInfo(program, this->device.get(), CL_PROGRAM_BUILD_LOG,
                              size, log, NULL);

        std::cout << "Build errors: \n" << log << std::endl;

        delete[] log;
        exit(build_stat);

      }

      cl_uint n_kernels = 0;
      cl_int kernel_creation_stat = clCreateKernelsInProgram(program, 0, NULL,
                                                             &n_kernels);

      if ((n_kernels > 0) && (kernel_creation_stat == CL_SUCCESS)) {
        cl_kernel k[n_kernels];
        clCreateKernelsInProgram(program, n_kernels, k, NULL);

        for (unsigned int i = 0; i < n_kernels; i++) {
          spKernel ik = spKernel(new Kernel(&this->device));
          ik->init(this->_cl, this->queue.get(), k[i]);
          this->kernels.insert(KernelMapPair(std::string(ik->name), ik));
        }

      } else {
        OpenCL::stat(kernel_creation_stat, "Could not get kernels: ");
        exit(kernel_creation_stat);
      }

      return this->kernels.size() - n_start_kernels;
    }

    Queue queue;
    Device device;

    std::vector<cl_device_id> devices;
    cl_context get() {
      return this->_cl;
    }
    cl_uint reference_count;
    cl_context_properties * properties;

  };

  static spContext setup(int device_index = -1, bool enable_profiling = false) {
    int overall_index = 1;
    cl_uint p;
    size_t d;
    // Indices are a pair<platform_index,device_index>
    cl_platform_id * platform_ids = NULL;
    cl_platform_id platform_id = NULL;
    cl_device_id device_id = NULL;
    std::shared_ptr<Context> context(new Context());
    Platform platform;
    cl_int status;

    // Get list of all devices, choose device and platform which goes with device.
    cl_uint n_platforms = 0;

    status = clGetPlatformIDs(0, NULL, &n_platforms);
    assert(status == CL_SUCCESS);

    platform_ids = new cl_platform_id[n_platforms];

    status = clGetPlatformIDs(n_platforms, platform_ids, NULL);
    assert(status == CL_SUCCESS);

    overall_index = 1;
    if (device_index == -1) {
      std::cout << "(0) Do not use OpenCL" << std::endl << std::flush;

      for (p = 0; p < n_platforms; p++) {
        platform.init(platform_ids[p]);
        std::cout << "Platform: " << platform.name << "(" << platform.vender
                  << ")" << std::endl;

        for (d = 0; d < platform.devices.size(); d++) {
          Device device;
          device.init(platform.devices[d]);
          std::cout << "\t(" << overall_index++ << ") " << device.name
                    << std::endl;
        }
        std::cout << std::flush;
      }

      std::cout << "Select device id: " << std::flush;
      std::string user_input;
      std::getline(std::cin, user_input);
      device_index = atoi(user_input.c_str());
      while (device_index < 0 || device_index >= overall_index) {
        std::cout << "Sorry, " << device_index << ", is not valid, try again"
                  << std::endl;
        user_input.assign("");
        std::getline(std::cin, user_input);
        device_index = atoi(user_input.c_str());
      }
    }

    if (device_index == 0) {

      std::cout << "Using single CPU thread" << std::endl << std::flush;
      return spContext(NULL);
    } else {

      overall_index = 0;

      for (p = 0; p < n_platforms; p++) {
        platform.init(platform_ids[p]);
        if (device_index <= (overall_index + int(platform.devices.size()))) {
          platform_id = platform_ids[p];
          device_id = platform.devices[device_index - overall_index - 1];
          break;
        } else {
          overall_index += platform.devices.size();
        }
      }

      context->init(platform_id, device_id,
                    (enable_profiling ? CL_QUEUE_PROFILING_ENABLE : 0));

      std::cout << "Using " << context->device.name << std::endl << std::flush;
    }
    delete[] platform_ids;

    return context;
  }

  static int stat(cl_int val, const char * msg) {
    std::string error_name = "None";
    if (val != CL_SUCCESS) {
      switch (val) {
        case CL_SUCCESS:
          error_name = "CL_SUCCESS";
          break;
        case CL_DEVICE_NOT_FOUND:
          error_name = "CL_DEVICE_NOT_FOUND";
          break;
        case CL_DEVICE_NOT_AVAILABLE:
          error_name = "CL_DEVICE_NOT_AVAILABLE";
          break;
        case CL_COMPILER_NOT_AVAILABLE:
          error_name = "CL_COMPILER_NOT_AVAILABLE";
          break;
        case CL_MEM_OBJECT_ALLOCATION_FAILURE:
          error_name = "CL_MEM_OBJECT_ALLOCATION_FAILURE";
          break;
        case CL_OUT_OF_RESOURCES:
          error_name = "CL_OUT_OF_RESOURCES";
          break;
        case CL_OUT_OF_HOST_MEMORY:
          error_name = "CL_OUT_OF_HOST_MEMORY";
          break;
        case CL_PROFILING_INFO_NOT_AVAILABLE:
          error_name = "CL_PROFILING_INFO_NOT_AVAILABLE";
          break;
        case CL_MEM_COPY_OVERLAP:
          error_name = "CL_MEM_COPY_OVERLAP";
          break;
        case CL_IMAGE_FORMAT_MISMATCH:
          error_name = "CL_IMAGE_FORMAT_MISMATCH";
          break;
        case CL_IMAGE_FORMAT_NOT_SUPPORTED:
          error_name = "CL_IMAGE_FORMAT_NOT_SUPPORTED";
          break;
        case CL_BUILD_PROGRAM_FAILURE:
          error_name = "CL_BUILD_PROGRAM_FAILURE";
          break;
        case CL_MAP_FAILURE:
          error_name = "CL_MAP_FAILURE";
          break;
        case CL_MISALIGNED_SUB_BUFFER_OFFSET:
          error_name = "CL_MISALIGNED_SUB_BUFFER_OFFSET";
          break;
        case CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST:
          error_name = "CL_EXEC_STATUS_ERROR_FOR_EVENTS_IN_WAIT_LIST";
          break;
        case CL_INVALID_VALUE:
          error_name = "CL_INVALID_VALUE";
          break;
        case CL_INVALID_DEVICE_TYPE:
          error_name = "CL_INVALID_DEVICE_TYPE";
          break;
        case CL_INVALID_PLATFORM:
          error_name = "CL_INVALID_PLATFORM";
          break;
        case CL_INVALID_DEVICE:
          error_name = "CL_INVALID_DEVICE";
          break;
        case CL_INVALID_CONTEXT:
          error_name = "CL_INVALID_CONTEXT";
          break;
        case CL_INVALID_QUEUE_PROPERTIES:
          error_name = "CL_INVALID_QUEUE_PROPERTIES";
          break;
        case CL_INVALID_COMMAND_QUEUE:
          error_name = "CL_INVALID_COMMAND_QUEUE";
          break;
        case CL_INVALID_HOST_PTR:
          error_name = "CL_INVALID_HOST_PTR";
          break;
        case CL_INVALID_MEM_OBJECT:
          error_name = "CL_INVALID_MEM_OBJECT";
          break;
        case CL_INVALID_IMAGE_FORMAT_DESCRIPTOR:
          error_name = "CL_INVALID_IMAGE_FORMAT_DESCRIPTOR";
          break;
        case CL_INVALID_IMAGE_SIZE:
          error_name = "CL_INVALID_IMAGE_SIZE";
          break;
        case CL_INVALID_SAMPLER:
          error_name = "CL_INVALID_SAMPLER";
          break;
        case CL_INVALID_BINARY:
          error_name = "CL_INVALID_BINARY";
          break;
        case CL_INVALID_BUILD_OPTIONS:
          error_name = "CL_INVALID_BUILD_OPTIONS";
          break;
        case CL_INVALID_PROGRAM:
          error_name = "CL_INVALID_PROGRAM";
          break;
        case CL_INVALID_PROGRAM_EXECUTABLE:
          error_name = "CL_INVALID_PROGRAM_EXECUTABLE";
          break;
        case CL_INVALID_KERNEL_NAME:
          error_name = "CL_INVALID_KERNEL_NAME";
          break;
        case CL_INVALID_KERNEL_DEFINITION:
          error_name = "CL_INVALID_KERNEL_DEFINITION";
          break;
        case CL_INVALID_KERNEL:
          error_name = "CL_INVALID_KERNEL";
          break;
        case CL_INVALID_ARG_INDEX:
          error_name = "CL_INVALID_ARG_INDEX";
          break;
        case CL_INVALID_ARG_VALUE:
          error_name = "CL_INVALID_ARG_VALUE";
          break;
        case CL_INVALID_ARG_SIZE:
          error_name = "CL_INVALID_ARG_SIZE";
          break;
        case CL_INVALID_KERNEL_ARGS:
          error_name = "CL_INVALID_KERNEL_ARGS";
          break;
        case CL_INVALID_WORK_DIMENSION:
          error_name = "CL_INVALID_WORK_DIMENSION";
          break;
        case CL_INVALID_WORK_GROUP_SIZE:
          error_name = "CL_INVALID_WORK_GROUP_SIZE";
          break;
        case CL_INVALID_WORK_ITEM_SIZE:
          error_name = "CL_INVALID_WORK_ITEM_SIZE";
          break;
        case CL_INVALID_GLOBAL_OFFSET:
          error_name = "CL_INVALID_LOCAL_OFFSET";
          break;
        case CL_INVALID_EVENT_WAIT_LIST:
          error_name = "CL_INVALID_EVENT_WAIT_LIST";
          break;
        case CL_INVALID_EVENT:
          error_name = "CL_INVALID_EVENT";
          break;
        case CL_INVALID_OPERATION:
          error_name = "CL_INVALID_OPERATION";
          break;
        case CL_INVALID_GL_OBJECT:
          error_name = "CL_INVALID_GL_OBJECT";
          break;
        case CL_INVALID_BUFFER_SIZE:
          error_name = "CL_INVALID_BUFFER_SIZE";
          break;
        case CL_INVALID_MIP_LEVEL:
          error_name = "CL_INVALID_MIP_LEVEL";
          break;
        case CL_INVALID_GLOBAL_WORK_SIZE:
          error_name = "CL_INVALID_LOCAL_WORK_SIZE";
          break;
        case CL_INVALID_PROPERTY:
          error_name = "CL_INVALID_PROPERTY";
          break;
      }
      std::cerr << error_name.c_str() << " " << msg << std::endl;
      throw OpenCL::Error(val, error_name.c_str(), msg);
      return 0;
    }

    return 1;
  }

  static size_t getWGMultiple(Kernel* k) {
    size_t wg_size;
    stat(clGetKernelWorkGroupInfo(k->get(), k->device->get(),
    CL_KERNEL_PREFERRED_WORK_GROUP_SIZE_MULTIPLE,
                                  sizeof(size_t), &wg_size, NULL),
         "Error getting kernel workgroup multiple");
    return wg_size;
  }

  static cl_int get_status(cl_event e) {
    cl_int estat;
    stat(
        clGetEventInfo(e, CL_EVENT_COMMAND_EXECUTION_STATUS, sizeof(cl_int),
                       &estat, NULL),
        "Error checking event status");
    return estat;
  }

  static CL_PROFILING_TIMES get_perf(cl_event event) {
    CL_PROFILING_TIMES times;
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_QUEUED,
                            sizeof(cl_ulong), &times.enqueued, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_SUBMIT,
                            sizeof(cl_ulong), &times.submitted, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(cl_ulong),
                            &times.started, NULL);
    clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(cl_ulong),
                            &times.ended, NULL);
    return times;
  }

};

class CLBacked {
 public:
  CLBacked(const OpenCL::spContext context)
      : context(context) {
  }

  CLBacked(CLBacked * other)
      : context(other->context) {
  }

  CLBacked(CLBacked& other)
      : context(other.context) {
  }

  const OpenCL::spContext context;
};

/**
 * Specializing _cl_setArg to handle lots of different types without needing to
 * convert argument types before calling setArgs
 */

#define CLMAP(t) t v=a; clSetKernelArg(this->_kernel, this->arg_idx++, sizeof(v), (void *) &v);

template<>
inline void OpenCL::Kernel::_cl_setArg<OpenCL::Memory&>(OpenCL::Memory& memory) {
  cl_mem v = memory.getCL();
  clSetKernelArg(this->_kernel, this->arg_idx++, sizeof(v), (void *) &v);
}


template<>
inline void OpenCL::Kernel::_cl_setArg(OpenCL::Memory * memory) {
  cl_mem v = memory->getCL();
  clSetKernelArg(this->_kernel, this->arg_idx++, sizeof(v), (void *) &v);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(OpenCL::KernelLocalMemory& mem) {
  clSetKernelArg(this->_kernel, this->arg_idx++, mem.size, NULL);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(OpenCL::KernelLocalMemory* mem) {
  clSetKernelArg(this->_kernel, this->arg_idx++, mem->size, NULL);
}


#ifdef half
template<>
inline void OpenCL::Kernel::_cl_setArg(half a) {
  CLMAP(cl_half);
}
#endif

template<>
inline void OpenCL::Kernel::_cl_setArg(bool a) {
  CLMAP(cl_char);
}

template<>

inline void OpenCL::Kernel::_cl_setArg(char a) {
  CLMAP(cl_char);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(unsigned char a) {
  CLMAP(cl_uchar);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(short a) {
  CLMAP(cl_short);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(unsigned short a) {
  CLMAP(cl_ushort);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(int a) {
  CLMAP(cl_int);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(unsigned int a) {
  CLMAP(cl_uint);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(long a) {
  CLMAP(cl_long);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(unsigned long a) {
  CLMAP(cl_ulong);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(float a) {
  CLMAP(cl_float);
}

template<>
inline void OpenCL::Kernel::_cl_setArg(double a) {
  CLMAP(cl_double);
}

#undef CLMAP

}
#endif // OPENCL_HPP
