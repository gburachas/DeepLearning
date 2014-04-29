#include <string>
#include <stdint.h>         // the normal place uint16_t is defined
#include <signal.h>
#include <iostream>
#include <fstream>
#include <gflags/gflags.h>

#include <ailab/common.h>
#include <ailab/algorithms/deepbeliefnet.h>
#include <ailab/algorithms/rbm.kernels.source.h>
#include <ailab/opencl_cli_mod.h>

DEFINE_bool(autosave, true,
            "Save weights and biases when complete or interupted");

DEFINE_bool(train, false, "Run training");
DEFINE_bool(reconstruct, false, "Reconstruct input");
DEFINE_string(energy, "", "Writes energy to this file if specified");

DEFINE_uint64(bufferLen, 100,
              "Number of rows to keep in the buffer for each input");
DEFINE_int32(batch_size, 0, "Non-zero will override the batch size");

DEFINE_bool(progress, true, "Log progress to the command line");

DEFINE_int32(logBins, 100,
             "Number of bins to use for histograms (equal width binning)");
DEFINE_int32(logFreq, 0,
             "Number of batches between logging lines (0 means never log)");
DEFINE_bool(logHistograms, false, "Include histograms in logs.");
DEFINE_bool(logError, true, "Include error in logs.");
DEFINE_bool(logErrorDetails, false, "Include row and column errors in logs.");
DEFINE_bool(logEnergy, false, "Include energy in logs.");

DEFINE_string(logFile, "", "Log file");
DEFINE_string(logURL, "", "Log to URL");

ailab::spDeepBeliefNet dbn(NULL);
ailab::DataIO<ailab::decimal_t> io;

void terminate(int p) {
  try {
    if (dbn != NULL) {
      dbn->save();
    }
    io.stop();
  } catch (...) {
    std::cerr << "Exception in terminate..." << std::endl;
  }
  exit(p);
}

int main(int argc, char * argv[]) {
  ailab::gen.seed( std::chrono::high_resolution_clock::now().time_since_epoch().count() );

#ifndef NDEBUG
  std::cerr << " === Running in debug mode === " << std::endl;
#endif

  google::ParseCommandLineFlags(&argc, &argv, true);
  ailab::OpenCL::spContext context = ailab::setup_opencl();

  if (FLAGS_autosave) {
    signal(SIGTERM, terminate);
    signal(SIGABRT, terminate);
    signal(SIGINT, terminate);
  }

  if (FLAGS_logFile.size()) {
    ailab::Logger::location.assign(FLAGS_logFile);
    ailab::Logger::factory = &ailab::Logger::initFS;
  } else if (FLAGS_logURL.size()) {
    ailab::Logger::location.assign(FLAGS_logURL);
    ailab::Logger::factory = &ailab::Logger::initHTTP;
  } else {
    ailab::Logger::factory = &ailab::Logger::initSTDOUT;
  }

  if (context != NULL) {
    context->loadKernels(ailab_rbm_kernels_source
                         , "-cl-denorms-are-zero -cl-single-precision-constant -cl-mad-enable -cl-fast-relaxed-math ");
  }

  dbn = ailab::spDeepBeliefNet(new ailab::DeepBeliefNet(context));

  io.set_buffer_len(FLAGS_bufferLen);

  if (argc > 1) {
    io.start();

    dbn->output_options.update_cli = FLAGS_progress;
    dbn->output_options.log_per_n_batches = FLAGS_logFreq;
    dbn->output_options.hist_bin_count = FLAGS_logBins;
    dbn->output_options.logHistograms = FLAGS_logHistograms;
    dbn->output_options.logError = FLAGS_logError;
    dbn->output_options.logErrorDetails = FLAGS_logErrorDetails;
    dbn->output_options.logEnergy = FLAGS_logEnergy;

    dbn->init(argv[1], io, FLAGS_batch_size);

    if (FLAGS_train) {
      dbn->train();
      dbn->save();
    }

    if (FLAGS_reconstruct) {
      dbn->reconstruct();
    }

    if (FLAGS_energy.length() > 0) {
      std::ofstream energyFile(FLAGS_energy);
      if (energyFile.is_open()) {
        dbn->write_energy(energyFile);
      } else {
        std::cerr << "Could not open energy file at " << FLAGS_energy;
      }
    }

    std::clog << "Flushing output..." << std::endl;
    io.stop();

  } else {
    std::cerr << "Please provide a .json configuration file" << std::endl;
  }
}
