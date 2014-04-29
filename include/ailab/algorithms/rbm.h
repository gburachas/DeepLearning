#ifndef RBM_H
#define RBM_H

#include <ailab/common.h>
#include <ailab/exception.h>
#include <ailab/oclmatrix.h>
#include <ailab/dataio.h>
#include <ailab/opencl.h>
#include <ailab/statcounter.h>
#include <ailab/algorithms/rbm.shared.units.h>
#include <ailab/clStructs.h>
#include <cstdint>
#include <functional>

#define RBM_STRUCT_VERSION 2

namespace ailab {

/**
 * @brief The RBM class
 *
 * Citation: Salakhutdinov, Ruslan, Andriy Mnih, and Geoffrey Hinton.
 *           "Restricted Boltzmann machines for collaborative filtering."
 *           ACM international conference proceeding series. Vol. 227. 2007.
 **/
class RBM : public CLBacked {
 public:

  class Layer;
  class UnitBlock;

  typedef std::shared_ptr<Layer> spLayer;
  typedef std::shared_ptr<UnitBlock> spUnitBlock;

  typedef std::vector<spUnitBlock> BlocksList;
  typedef OCLMatrix<decimal_t> Matrix;

  typedef std::shared_ptr<Matrix> spMatrix;
  typedef std::pair<RBMApplyFunc, OpenCL::spKernel> FuncApplyPair;

  typedef std::function<bool(Matrix&)> Spout;
  typedef std::function<void(Matrix&)> Drain;

  typedef struct output_t {
    bool update_cli;
    bool logHistograms;
    bool logEnergy;
    bool logError;
    bool logErrorDetails;
    bool logEntropy;
    bool logEffect;
    size_t log_per_n_batches;
    size_t hist_bin_count;
  } OutputOptions;

  typedef struct weights_header_t {
      uint8_t version;
      uint8_t weight_bytes;
      uint8_t dual_matrices;
      uint8_t has_momentum;
      uint32_t n_updates;
      uint32_t nvis;
      uint32_t nhid;
  } WeightsHeader;

  typedef struct layer_diff_t{
      double data;
      double recon;
      double delta;
  } LayerDiff;

  /*----------------------------------------
   Public Member Classes
   ----------------------------------------*/

  /**
   Homogeneous Block of Units which can be processed in a data-parallel manner
   **/
  class UnitBlock : public CLBacked {
   protected:
    OpenCL::spKernel kGen;
    OpenCL::spKernel kSample;

    Layer* layer;
    UnitConfig& cfg;

    OpenCL::spKernel get_kernel(std::string fmt);
    std::map<std::string, FuncApplyPair> funcsToApply;
   public:
    UnitBlock(Layer * layer, std::string type, size_t start_index,
              size_t number_of_units,
              std::vector<decimal_t>& params,
              std::vector<decimal_t>& unit_params,
              const OpenCL::spContext context);

    void set_layer(Layer* to);

    static bool islessthan(UnitBlock* a, UnitBlock* b);
    static bool islessthan(UnitBlock& a, UnitBlock& b);

    virtual void sample(Matrix& input,
                        Matrix& randoms,
                        Matrix& output,
                        RBMParams& params);

    virtual void gen(Matrix& input,
                     Matrix& output,
                     Matrix& weights,
                     RBMParams& params);
    virtual void apply(Matrix& data, std::string func_name, RBMParams& params);

    std::string type;
    size_t param_count;
    size_t start_index;
    size_t number_of_units;
    size_t on_sample_number;

    bool can_be_sampled;
    Matrix parameters;
    OCLMatrix<rbm_unit_params_t> unit_parameters;
    OCLMatrix<rbm_unit_state_t> unit_state;
  };

  class Layer : public CLBacked {
   protected:
    void apply(Matrix& to, std::vector<std::string>& functions,
               RBMParams& params);
    void gen(Matrix &my_data, Matrix &their_data, Matrix& weights,
             RBMParams& params);
    void norm_cols(Matrix& data);

    bool _initialized;

    OpenCL::spKernel kUpdateBias;
    OpenCL::spKernel kNormByCol;
    OpenCL::spKernel kRowError;
    OpenCL::spKernel kColError;
    OpenCL::spKernel kEnergyI;
    OpenCL::spKernel kEnergyJ;
    OpenCL::spKernel kEnergyFinal;    
    OpenCL::spKernel kEntropy;
    OpenCL::spKernel kGradientReconstruct;
    OpenCL::spKernel kMaskingGradientReconstruct;

   public:
    Layer(const OpenCL::spContext context)
        : apply_range(context),
          biases(context),
          biasExp(context),
          data(context),
          dataSample(context),
          sampleProbs(context),
          energy(context),
          energyPartial(context),
          entropyPartial(context),
          col_error(context),
          recon(context),
          row_error(context),
          is_visible(false),
          _initialized(false),
          CLBacked(context) {

      if (context != NULL) {
        this->kRowError = context->getKernel("set_row_error");
        this->kColError = context->getKernel("set_col_error");
        this->kNormByCol = context->getKernel("norm_by_column");
        this->kUpdateBias = context->getKernel("update_bias");
        this->kEnergyI = context->getKernel("energy_i");
        this->kEnergyJ = context->getKernel("energy_j");
        this->kEnergyFinal = context->getKernel("energy_final");
        this->kEntropy = context->getKernel("layer_entropy");
        this->kGradientReconstruct = context->getKernel("gradient_reconstruct");
        this->kMaskingGradientReconstruct = context->getKernel("masking_gradient_reconstruct");
      }

    }

    void init(size_t total_width, size_t batch_size,
              OCLMatrix<uint>* rng_states);

    virtual bool init_blocks_json(json_value* def, OCLMatrix<uint>* rng_states,
                                  size_t batch_size);

    virtual bool init_blocks_vec(std::vector<spUnitBlock>& blocks,
                                 OCLMatrix<uint>* rng_states,
                                 size_t batch_size);

    void gen_recon(Matrix &input, Matrix& weights, RBMParams& params);
    void gen_data(Matrix &input, Matrix& weights, RBMParams& params);

    size_t gradient_reconstruct(size_t rowNumber, zbin::SelectBox& section
                                , Matrix& mask, OCLMatrix<cl_uint>& maskSum, RBMParams& params);

    void gradient_reconstruct(size_t rowNumber, zbin::SelectBox& section, RBMParams& params);

    void sample(Matrix &input, Matrix &output, RBMParams& params);

    // This is from the perspective that "this" is a
    // visible layer
    void set_energy(bool forData, spLayer hidden, Matrix& weights, RBMParams& params);

    void apply_to_data(std::vector<std::string>& functions, RBMParams& params);
    void apply_to_recon(std::vector<std::string>& functions, RBMParams& params);

    bool isInitialized();

    virtual void update_bias(RBMParams& params);
    void norm_data_cols();
    void norm_recon_cols();
    void set_error();

    virtual void set_row_error();
    virtual void set_row_error(ailab::StatCounter * counters);
    virtual void set_column_error();
    virtual void set_column_error(ailab::StatCounter * counters);

    LayerDiff get_entropy();

    Matrix data;
    Matrix dataSample;
    Matrix sampleProbs;
    Matrix recon;
    Matrix biases;
    Matrix biasExp;
    Matrix energy;
    Matrix energyPartial;
    Matrix entropyPartial;
    Matrix apply_range;
    Matrix col_error;
    Matrix row_error;

    // This is a pointer because rng_states are shared amount everybody
    // And is guaranteed to be at least as wide as the largest layer
    OCLMatrix<uint>* rng_states;
    bool is_visible;
    std::vector<spUnitBlock> blocks;
  };

  /** ------------------------------
   *      Constructor
   *------------------------------ **/
  RBM(const OpenCL::spContext context);
  ~RBM() {
  }

  /** ------------------------------ Initialization ------------------------------ **/

  bool init(std::string json_file, size_t with_batch_size);
  bool init(std::string json_file, size_t with_batch_size, std::string& name);

  bool init(json_value* json, std::string& directory,
            size_t with_batch_size, std::string& name,
            spLayer visible);

  bool init_with_visible(size_t visible_width);

  void init_set_params(json_value* root, size_t with_batch_size);

  Matrix& weights(bool get_vxh);
  Matrix& velocity(bool get_vxh);

  void fillRand();

  /** ------------------------------ Members ------------------------------ **/
  void feed_forward();
  void feed_backward();

  /** ------------ Train the layer using Contrastive Divergence --------------- **/
  void run(bool genHidden);
  void run(Matrix& hiddenData);

  void gibbs_sample(size_t gibbs);

  void train(Spout dataSpout, Spout holdoutSpout);
  void cross_train(Spout dataSpout, Spout featureSpout, Spout dataHoldoutSpout,
                   Spout featureHoldoutSpout);

  void cross_train(RBM& crossModel, Spout myData, Spout crossData);

  void reconstruct(zbin::SelectBox& section, size_t iterations, Spout dataSpout, Drain dataSink);
  void calc_energy(Spout dataSpout, Drain energySink);
  void print_col_error(Spout input, Drain output);
  void print_row_error(Spout input, Drain output);

  void gen_from_features(Spout featureSpout, Drain dataSink);
  void gen_features(Spout dataSpout, Drain featureSink);

  void cross_gen(size_t iters, zbin::SelectBox& section, RBM& crossModel, Spout dataInput, Drain dataOutput);

  void trainingCLILog(size_t batch, StatCounter& data_error, StatCounter& holdout_error);

  void trainingLog(size_t batch, StatCounter& data_error, StatCounter& holdout_error);

  /***
   *  Logging and informational functions
   ***/

  /**
   *	Using mean absolute error
   */
  decimal_t error();
  decimal_t errStdev();
  decimal_t errMean();
  std::vector<decimal_t> unit_error();
  void logSetup(unsigned int epochs);
  void logEpoch(unsigned int e, unsigned int epoch_count);
  void logHistograms(size_t nbins, bool backprop = false);
  void logFreeEnergy();

  void logError(decimal_t d, decimal_t h =
                    std::numeric_limits<decimal_t>::quiet_NaN(),
                bool backprop = false);

  void logMeanError(StatCounter& data_error,
                    StatCounter& holdout_error,
                    bool backprop = false);

  void logEffect();
  void logEntropy(LayerDiff &diff);

  /***
   * Updating and misc functions
   ***/

  void update();
  bool weights_are_symmetric();
  void untie_weights();

  size_t nvis();
  size_t nhid();

  Matrix& get_visible_energy(bool forData);


  /**
   * @brief load Load previously saved weights and biases,
   *          the exact file name is the name of the {RBM Name}.weights
   * @param directory
   * @return True if load was successful
   */
  bool load(std::string directory);

  /**
   * @brief save Save the weights and biases to a binary file named {RBM Name}.weights
   * @param directory
   * @return True of save was successful
   */
  bool save();

  bool isSetup();

  bool verify_ready();

  bool shouldKeepTraining();
  decimal_t getEffectSize();

  bool is_associative_memory;
  bool fixed_weights;

  RBMParams params;

  size_t epochs;
  size_t maxTrainingUpdates;
  size_t minUpdates;
  decimal_t maxEffectSize;
  StatCounter dataEntropy;
  StatCounter reconEntropy;

  bool useChurn;

  Matrix weights_vxh;
  Matrix weights_hxv;

  Matrix velocity_vxh;
  Matrix velocity_hxv;

  Matrix kernelReturn;

  spLayer visible;
  spLayer hidden;

  OCLMatrix<uint> rng_states;
  uint4 * rand_state;

  size_t updatesCnt;
  std::string name;
  std::string rootdir;
  std::string json_path;

  OutputOptions output_options;

 protected:
  OpenCL::spKernel kUpdateWeights;
  OpenCL::spKernel kWeightPenalty;

  bool using_momentum;
  bool feeding_forward;
  bool init_core();

  decimal_t get_weight_penalty();
  decimal_t prevWeightPenalty;
  std::atomic_bool wasUpdated;

  bool cl_update_weights();
  void update_weights();
  void log(std::stringstream &io);

  uint32_t n_updates;
  size_t batch_counter;
  size_t epoch_counter;
  bool _loaded_weights;

  struct {
    double oldMean;
    double newMean;
    double oldSum;
    double newSum;
    size_t count;
  } errorStats;

};

typedef std::shared_ptr<RBM> spRBM;

}

#endif // RBM_HPP
