#include <ailab/common.h>
#include <ailab/oclmatrix.h>
#include <ailab/opencl.h>
#include <vector>
#include <list>
#include <map>
#include <random>
#include <string.h>

#include <libgen.h>
#include <ext/json.h>

#include <iostream>
#include <iomanip>

#include <ailab/oclmatrix.h>
#include <ailab/algorithms/rbm.h>
#include <ailab/exception.h>
#include <ailab/dataio.h>
#include <ailab/statcounter.h>

#include <ailab/clStructs.h>

namespace ailab {

    // =======================================================================================
    // RBM::UnitBlock
    // =======================================================================================

    RBM::UnitBlock::UnitBlock(ailab::RBM::Layer *layer, std::string type,
            size_t start_index, size_t number_of_units,
            std::vector<decimal_t> &params,
            std::vector<decimal_t> &unit_params,
            const OpenCL::spContext context)
    : layer(layer),
    type(type),
    cfg(global_rbm_all_units_config[type]),
    start_index(start_index),
    number_of_units(number_of_units),
    parameters(context),
    unit_parameters(context),
    unit_state(context),
    CLBacked(context) {
        /**
         * Basic Assertions
         */
        assert(global_rbm_all_units_config[type].param_count == params.size());
        uint i = 0;

        UnitConfig& cfg = global_rbm_all_units_config[type];
        this->param_count = cfg.param_count;
        this->can_be_sampled = cfg.can_be_sampled;

        this->parameters.init({std::max(this->param_count, params.size())});
        assert(params.size() == this->parameters.n_cells);
        this->parameters.read_from(params);

        this->unit_parameters.init({number_of_units, cfg.unit_param_count});
        assert(unit_params.size() == this->unit_parameters.n_cells);
        auto piter = unit_params.begin();
        auto n_unit_params = this->unit_parameters.rows();
        for (i = 0; i < n_unit_params; i++) {
            auto p = this->unit_parameters.at(i);
            p.w = *piter++;
            p.x = *piter++;
            p.y = *piter++;
            p.z = *piter++;
        }

        this->unit_state.init({number_of_units, 1});
        this->unit_state.set({0.0, 0.0});

        if (this->context != nullptr) {
            this->kSample = this->get_kernel("sample_%s_unit");
            this->kGen = this->get_kernel("generate_%s_unit");
        }
    }

    void RBM::UnitBlock::set_layer(RBM::Layer *to) {
        this->layer = to;
    }

    // Used for sorting

    bool RBM::UnitBlock::islessthan(UnitBlock* a, UnitBlock* b) {
        return a->type < b->type;
    }

    bool RBM::UnitBlock::islessthan(UnitBlock& a, UnitBlock& b) {
        return a.type < b.type;
    }

    OpenCL::spKernel RBM::UnitBlock::get_kernel(std::string fmt) {
        return this->context->getKernel(global_rbm_all_units_config[this->type].name, fmt.c_str());
    }

    /**
     * @brief RBM::UnitBlock::gen
     * @param input
     * @param output
     * @param weights Shaped as Rows x Columns => input x output
     */
    void RBM::UnitBlock::gen(Matrix& input, Matrix& output, Matrix& weights,
            RBMParams& params) {

        assert(weights.rows() == input.cols());
        assert(weights.cols() == output.cols());

        if (this->kGen != NULL) {

            OpenCL::KernelLocalMemory cacheRow(sizeof (cl_float) * input.cols());

            this->kGen->set_size(this->number_of_units, output.rows());
            this->kGen->set_offset(this->start_index, 0);
            this->kGen->run(input.getCLConst(),
                    output.getCL(),
                    this->layer->biases.getCLConst(),
                    this->layer->biasExp.getCLConst(),
                    weights.getCLConst(),
                    this->unit_state.getCL(),
                    this->unit_parameters.getCLConst(),
                    this->parameters.getCL(),
                    &cacheRow,
                    this->layer->is_visible,
                    params);

        } else {

            const bool use_sparsity = (params.sparsityTarget > 0) && !this->layer->is_visible;
            Matrix& biases = this->layer->biases;
            Matrix& biasExp = this->layer->biases;
            auto typeParams = this->parameters.getHostConst();
            this->unit_parameters.getHostConst();
            this->unit_state.getHostConst();

            const auto inner = this->cfg.inner_sum;
            const auto activation = this->cfg.activation;

            const auto endUnit = this->start_index + this->number_of_units;
            const auto input_width = input.cols();

            this->layer->biases.getHostConst();
            biases.getHostConst();
            biasExp.getHostConst();

            for (auto unit = this->start_index; unit < endUnit; unit++) {
                decimal2 * myState = this->unit_state.n_cells < unit ? this->unit_state.ref(unit) : NULL;
                decimal4 * myParams = this->unit_parameters.n_cells < unit ? this->unit_parameters.ref(unit) : NULL;
                decimal_t b = biases.at(unit);
                decimal_t& unitExp = biasExp.at(unit);

                for (size_t r = 0; r < output.rows(); r++) {
                    decimal_t z = 0;

                    for (size_t i = 0; i < input_width; i++) {
                        z += inner(input.at(r, i), weights.at(i, unit), myState, myParams, typeParams);
                    }

                    if (use_sparsity) {
                        z += (params.sparsityTarget - unitExp) * params.sparsityCost;
                    }

                    output.at(r, unit) = activation(z, b, myState, myParams, typeParams);
                    assert(std::isfinite(output.at(r, unit)));
                }

            }

        }
    }

    void RBM::UnitBlock::sample(Matrix& input, Matrix& randoms, Matrix& output,
            RBMParams& params) {
        if (this->cfg.can_be_sampled) {
            // Actually Sample
            if (this->kSample != NULL) {
                this->kSample->set_size(this->number_of_units, output.rows());
                this->kSample->set_offset(this->start_index, 0);
                this->kSample->run(input.getCLConst(), randoms.getCLConst(),
                        output.getCL(), static_cast<cl_uint> (output.cols()),
                        params);

            } else {

                output.getHostConst();
                randoms.getHostConst();
                input.getHost();

                for (size_t i = 0; i < output.n_cells; i++)
                    output.at(i) = 1.0 * (input.at(i) > randoms.at(i));
            }
        }
    }

    void RBM::UnitBlock::apply(Matrix& data, std::string func_name,
            RBMParams& params) {
        auto fIter = this->funcsToApply.find(func_name);
        if (fIter != this->funcsToApply.end()) {

            RBM::FuncApplyPair& applyPair = fIter->second;

            this->unit_parameters.getHostConst();
            this->unit_state.getHostConst();

            if (applyPair.second == NULL) {
                auto typeParams = this->parameters.getHostConst();
                auto& biases = this->layer->biases;
                auto& dest = this->layer->apply_range;
                size_t end_index = this->start_index + this->number_of_units;

                data.getHostConst();
                biases.getHostConst();
                dest.getHost();

                for (size_t c = this->start_index; c < end_index; c++) {

                    rbm_unit_state_t * c_state = (c < this->unit_state.n_cells) ? this->unit_state.ref(c) : NULL;
                    rbm_unit_params_t * c_params = (c < this->unit_parameters.n_cells) ? this->unit_parameters.ref(c) : NULL;

                    for (size_t r = 0; r < data.rows(); r++) {
                        dest.at(r, c) = applyPair.first(data.at(r, c), biases.at(c), c_state, c_params, typeParams);
                    }
                }

            } else {
                OpenCL::spKernel k = applyPair.second;
                k->set_size(this->number_of_units, data.rows());
                k->set_offset(this->start_index);
                k->run(data.getCL(), this->layer->apply_range.getCL(),
                        this->layer->biases.getCLConst(), this->parameters.getCLConst(),
                        params, static_cast<cl_uint> (data.cols()));
            }
        }
    }

    // =======================================================================================
    // RBM::Layer
    // =======================================================================================

    void RBM::Layer::init(size_t total_width, size_t batch_size,
            OCLMatrix<uint> *rng_states) {

        this->kUpdateBias = this->context == nullptr ? NULL : this->context->getKernel("update_bias");
        this->apply_range.init({batch_size, total_width});
        this->biases.init({1, total_width});
        this->biasExp.init({1, total_width});
        this->col_error.init({1, total_width});
        this->data.init({batch_size, total_width});
        this->dataSample.init({batch_size, total_width});
        this->sampleProbs.init({batch_size, total_width});
        this->energy.init({batch_size, 1});
        this->energyPartial.init({batch_size, total_width});
        this->entropyPartial.init({2, total_width});
        this->recon.init({batch_size, total_width});
        this->rng_states = rng_states;
        this->row_error.init({1, batch_size});
        this->_initialized = true;
        
        assert((batch_size * total_width) > 0);

        if (this->blocks.size() == 0) {
            std::vector<decimal_t> dummy_vec;
            std::vector<unsigned int> tied_units;

            this->blocks.push_back(
                    ailab::RBM::spUnitBlock(
                    new ailab::RBM::UnitBlock(this, "binary", 0, total_width, dummy_vec, dummy_vec,
                    this->context)));
        }

    }

    bool RBM::Layer::isInitialized() {
        return this->_initialized;
    }

    bool RBM::Layer::init_blocks_json(json_value *def, OCLMatrix<uint> *rng_states,
            size_t batch_size) {
        assert(def->type == JSON_ARRAY);
        size_t layer_width = 0;
        DVec b_params;
        DVec u_params;

        this->blocks.clear();

        for (json_value *it = def->first_child; it; it = it->next_sibling) {
            assert(it->type == JSON_OBJECT);

            std::string v_type = "binary";
            size_t width = 100;

            for (json_value *propIter = it->first_child; propIter;
                    propIter = propIter->next_sibling) {

                if (propIter->name != NULL) {
                    if (strcmp(propIter->name, "type") == 0) {
                        assert(propIter->type == JSON_STRING);
                        v_type.assign(propIter->string_value);
                    } else if (strcmp(propIter->name, "width") == 0) {
                        assert(propIter->type == JSON_INT);
                        width = propIter->int_value;
                    } else if (strcmp(propIter->name, "parameters") == 0) {
                        assert(propIter->type == JSON_ARRAY);
                        for (json_value *pIter = propIter->first_child; pIter;
                                pIter = pIter->next_sibling) {
                            assert(pIter->type == JSON_FLOAT);
                            b_params.push_back(pIter->float_value);
                        }

                    } else if (strcmp(propIter->name, "unit_parameters") == 0) {
                        assert(propIter->type == JSON_ARRAY);
                        for (json_value *pIter = propIter->first_child; pIter;
                                pIter = pIter->next_sibling) {
                            assert(pIter->type == JSON_FLOAT);
                            u_params.push_back(pIter->float_value);
                        }

                    }
                }

            }

            this->blocks.push_back(
                    ailab::RBM::spUnitBlock(
                    new ailab::RBM::UnitBlock(this, v_type, layer_width, width,
                    b_params, u_params, context)));
            layer_width += width;
        }

        this->init(layer_width, batch_size, rng_states);

        return true;
    }

    bool RBM::Layer::init_blocks_vec(std::vector<spUnitBlock> &blocks,
            OCLMatrix<uint>* rng_states,
            size_t batch_size) {
        size_t layer_width = 0;
        this->blocks.clear();
        for (RBM::spUnitBlock b : blocks) {
            this->blocks.push_back(b);
            layer_width += b->number_of_units;
        }
        this->init(layer_width, batch_size, rng_states);

        return layer_width > 0;
    }

    void RBM::Layer::gen_recon(Matrix& input, Matrix& weights, RBMParams& params) {
        this->gen(this->recon, input, weights, params);
    }

    void RBM::Layer::gen_data(RBM::Matrix &input, Matrix& weights,
            RBMParams& params) {
        this->gen(this->data, input, weights, params);
    }

    void RBM::Layer::gen(RBM::Matrix &my_data, RBM::Matrix &their_data,
            Matrix& weights, RBMParams& params) {
        for (RBM::spUnitBlock b : blocks)
            b->gen(their_data, my_data, weights, params);
    }

    void RBM::Layer::sample(Matrix &input, Matrix &output, RBMParams& params) {

        this->sampleProbs.fillRandUniform(0.0, 1.0);

        for (RBM::spUnitBlock b : blocks) {
            b->sample(input, this->sampleProbs, output, params);
        }
    }

    size_t RBM::Layer::gradient_reconstruct(size_t rowNumber,
            zbin::SelectBox& section, Matrix& mask,
            OCLMatrix<cl_uint>& maskSum,
            RBMParams& params) {

        size_t rowOffset =
                (section.startRow > rowNumber) ? section.startRow - rowNumber : 0;
        size_t nrows = std::min(this->data.rows(), (section.endRow - rowNumber))
                - rowOffset;
        size_t ncols = section.endCol - section.startCol;
        cl_uint pitch = this->data.cols();

        if (this->kMaskingGradientReconstruct != NULL) {
            auto k = this->kMaskingGradientReconstruct;
            k->set_size(nrows, ncols);
            k->set_offset(rowOffset, section.startCol);

            maskSum.set(0);
            k->run(this->data.getCL(), this->recon.getCLConst(), mask.getCL(),
                    maskSum.getCL(), params.epsilon, pitch);

            maskSum.getHost();
            return maskSum.at(0);

        } else {
            std::cerr << "Sorry, not implemented on the CPU at the moment."
                    << std::endl;
            return 0;
        }
    }

    void RBM::Layer::gradient_reconstruct(size_t rowNumber,
            zbin::SelectBox& section,
            RBMParams& params) {

        size_t rowOffset =
                (section.startRow > rowNumber) ? section.startRow - rowNumber : 0;
        size_t nrows = std::min(this->data.rows(), (section.endRow - rowNumber))
                - rowOffset;
        size_t ncols = section.endCol - section.startCol;
        cl_uint pitch = this->data.cols();

        if (this->kGradientReconstruct != NULL) {
            auto k = this->kGradientReconstruct;
            k->set_size(nrows, ncols);
            k->set_offset(rowOffset, section.startCol);
            k->run(this->data.getCL(), this->recon.getCLConst(), params.epsilon, pitch);
        } else {
            std::cerr << "Sorry, not implemented on the CPU at the moment."
                    << std::endl;
        }
    }

    void RBM::Layer::set_energy(bool forData, spLayer hidden, Matrix& weights,
            RBMParams& params) {

        Matrix& vis = (forData) ? this->data : this->recon;

        if ((this->kEnergyFinal != NULL) && (this->kEnergyI != NULL)
                && (this->kEnergyJ != NULL)) {

            assert(this->energyPartial.getCL() != NULL);
            assert(hidden->energyPartial.getCL() != NULL);
            assert(this->energy.getCL() != NULL);
            assert(weights.getCLConst() != NULL);

            this->kEnergyI->set_size(params.nvis);
            this->kEnergyI->run(this->energyPartial.getCL()
                                , vis.getCLConst()
                                , this->biases.getCLConst()
                                , params);

            this->kEnergyJ->set_size(params.nhid);
            this->kEnergyJ->run(hidden->energyPartial.getCL()
                                , vis.getCLConst()
                                , hidden->biases.getCLConst()
                                , weights.getCLConst()
                                , params);

            this->kEnergyFinal->set_size(params.batchSize);
            this->kEnergyFinal->run(this->energy.getCL()
                                    , this->energyPartial.getCLConst()
                                    , hidden->energyPartial.getCLConst()
                                    , params);

        } else {
            this->energy.getHost();

            Matrix& hid_bias = hidden->biases;
            Matrix& vis = (forData) ? this->data : this->recon;
            Matrix& vis_bias = this->biases;

            decimal_t sumj = 0.0;
            decimal_t sumi = 0.0;
            decimal_t xj;
            uint r, i, j;

            for (r = 0; r < params.batchSize; r++) {
                sumi = 0.0;
                for (i = 0; i < params.nvis; i++) {
                    sumi += vis.at(r, i) * vis_bias.at(i);
                }

                sumj = 0.0;
                for (j = 0; j < params.nhid; j++) {
                    xj = hid_bias.at(j);
                    for (i = 0; i < params.nvis; i++) {
                        xj += vis.at(r, i) * weights.at(i, j);
                    }
                    sumj += ::log(1 + exp(xj));
                }

                this->energy.at(r) = -sumi - sumj;
            }
        }
    }

    void RBM::Layer::apply_to_data(std::vector<std::string> &functions,
            RBMParams& params) {
        this->apply(this->data, functions, params);
    }

    void RBM::Layer::apply_to_recon(std::vector<std::string> &functions,
            RBMParams& params) {
        this->apply(this->recon, functions, params);
    }

    void RBM::Layer::apply(Matrix& to, std::vector<std::string> &functions,
            RBMParams& params) {
        for (std::string fname : functions) {
            for (RBM::spUnitBlock b : blocks) {
                b->apply(to, fname, params);
            }
        }
    }

    void RBM::Layer::update_bias(RBMParams& params) {
        cl_uint pitch = this->biases.cols();

        if (this->kUpdateBias == NULL) {
            const bool use_sparsity = params.sparsityTarget > 0 && !this->is_visible;
            this->data.getHostConst();
            this->recon.getHostConst();
            this->biases.getHost();
            this->biasExp.getHost();

            for (size_t u = 0; u < pitch; u++) {
                decimal_t& bStat = this->biasExp.at(u);
                decimal_t on_data = 0.0;
                decimal_t on_recon = 0;

                for (size_t r = 0; r < params.batchSize; r++) {
                    on_data += this->data.at(r, u);
                    on_recon += this->recon.at(r, u);
                }

                if (use_sparsity) {
                    bStat = (bStat * params.biasDecay) + ((1.0 - params.biasDecay) * (on_recon / params.batchSize));
                }

                this->biases.at(u) += params.epsilonDivBatch * (on_data - on_recon);
            }

        } else {

            assert(this->biases.getCL() != NULL);
            assert(this->biasExp.getCL() != NULL);
            assert(this->data.getCL() != NULL);
            assert(this->recon.getCL() != NULL);

            this->kUpdateBias->set_size(this->biases.cols());
            this->kUpdateBias->set_offset(0);
            this->kUpdateBias->run(this->biases.getCL(), this->biasExp.getCL(),
                    this->data.getCLConst(), this->recon.getCLConst(),
                    this->is_visible, params, pitch);
        }
    }

    void RBM::Layer::norm_data_cols() {
        this->norm_cols(this->data);
    }

    void RBM::Layer::norm_recon_cols() {
        this->norm_cols(this->recon);
    }

    void RBM::Layer::norm_cols(Matrix& data) {
        if (this->kNormByCol == NULL) {
            data.getHost();
            size_t u, r;
            for (u = 0; u < data.cols(); u++) {
                decimal_t factor = 0.0;
                for (r = 0; r < data.rows(); r++)
                    factor += data.at(r, u);
                factor = 1.0 / factor;
                for (r = 0; r < data.rows(); r++)
                    data.at(r, u) *= factor;
            }

        } else {
            this->kNormByCol->set_size(data.cols());
            this->kNormByCol->set_offset(0);
            this->kNormByCol->run(data.getCL(), static_cast<cl_uint> (data.cols()),
                    static_cast<cl_uint> (data.rows()));
        }
    }

    void RBM::Layer::set_error() {
        this->set_row_error();
        this->set_column_error();
    }

    void RBM::Layer::set_row_error() {
        if (this->kRowError == NULL) {
            this->data.getHostConst();
            this->recon.getHostConst();
            this->row_error.getHost();

            decimal_t total_error;

            for (size_t r = 0; r < data.rows(); r++) {
                total_error = 0.0;
                for (size_t i = 0; i < data.cols(); i++) {
                    total_error += fabs(this->data.at(r, i) - this->recon.at(r, i));
                }
                this->row_error.at(r) = total_error / data.cols();
            }

        } else {
            this->kRowError->set_size(this->data.rows());
            this->kRowError->set_offset(0);
            this->kRowError->run(this->data.getCLConst(), this->recon.getCLConst(),
                    this->row_error.getCL(),
                    static_cast<cl_uint> (this->data.cols()),
                    static_cast<cl_uint> (this->data.rows()));
        }
    }

    void RBM::Layer::set_row_error(ailab::StatCounter * counters) {

        this->data.getHostConst();
        this->recon.getHostConst();

        for (size_t r = 0; r < data.rows(); r++) {
            StatCounter& c = counters[r];
            c.reset();
            for (size_t i = 0; i < data.cols(); i++) {
                c.push(fabs(this->data.at(r, i) - this->recon.at(r, i)));
            }
        }
    }

    void RBM::Layer::set_column_error(ailab::StatCounter * counters) {

        this->data.getHostConst();
        this->recon.getHostConst();

        for (size_t i = 0; i < data.cols(); i++) {
            StatCounter& c = counters[i];
            for (size_t r = 0; r < data.rows(); r++) {
                c.push(fabs(this->data.at(r, i) - this->recon.at(r, i)));
            }
        }
    }

    void RBM::Layer::set_column_error() {
        if (this->kColError == NULL) {
            this->data.getHostConst();
            this->recon.getHostConst();
            this->col_error.getHost();

            decimal_t total_error;

            for (size_t i = 0; i < data.cols(); i++) {
                total_error = 0.0;
                for (size_t r = 0; r < data.rows(); r++) {
                    total_error += fabs(this->data.at(r, i) - this->recon.at(r, i));
                }

                this->col_error.at(i) = total_error / data.rows();
            }

        } else {
            this->kColError->set_size(this->data.cols());
            this->kColError->set_offset(0);
            this->kColError->run(this->data.getCLConst(), this->recon.getCLConst(),
                    this->col_error.getCL(),
                    static_cast<cl_uint> (this->data.cols()),
                    static_cast<cl_uint> (this->data.rows()));
        }
    }

    RBM::LayerDiff RBM::Layer::get_entropy(){
        LayerDiff ent = {0,0,0};
        auto cols = this->data.cols();

        this->kEntropy->set_size( cols );
        this->kEntropy->set_offset(0);
        this->kEntropy->run(
                    this->entropyPartial.getCL()
                    , this->data.getCLConst()
                    , this->recon.getCLConst()
                    , static_cast<cl_uint> (this->data.cols())
                    , static_cast<cl_uint> (this->data.rows()) );

        this->entropyPartial.getHostConst();

        for(size_t c=0; c < cols; c++)
        {
            ent.data += this->entropyPartial.at(0, c);
            ent.recon += this->entropyPartial.at(1, c);
        }

        ent.data = -ent.data;
        ent.recon = -ent.recon;
        ent.delta = ent.data - ent.recon;
        return ent;
    }

    // =======================================================================================
    /**
     * @brief The RBM class
     *
     * Citation: Salakhutdinov, Ruslan, Andriy Mnih, and Geoffrey Hinton. "Restricted Boltzmann machines for collaborative filtering." ACM international conference proceeding series. Vol. 227. 2007.
     */

    /**
     * @brief RBM::RBM OpenCL Constructor
     * @param context
     */
    RBM::RBM(const OpenCL::spContext context)
    : fixed_weights(false),
    feeding_forward(true),
    using_momentum(false),
    batch_counter(0),
    wasUpdated(false),
    useChurn(true),
    prevWeightPenalty(0),
    _loaded_weights(false),
    weights_vxh(context),
    weights_hxv(context),
    velocity_vxh(context),
    velocity_hxv(context),
    kernelReturn(context),
    rng_states(context),
    kUpdateWeights(NULL),
    epochs(0),
    n_updates(0),
    maxTrainingUpdates(0),
    minUpdates(0),
    maxEffectSize(0.0),
    CLBacked(context) {
        memset((char*) & this->params, 0, sizeof (this->params));

        this->params.batchSize = 10;
        this->params.epsilon = 0.05;
        this->params.gibbs = 1;
        this->params.statLen = 100;

        this->output_options.update_cli = false;
        this->output_options.log_per_n_batches = 0;
        this->output_options.hist_bin_count = 100;

        this->output_options.logHistograms = false;
        this->output_options.logError = true;
        this->output_options.logEnergy = false;

        this->errorStats = {0, 0, 0, 0, 0};
    }

    /** ------------------------------ Initialization ------------------------------ **/
    bool RBM::init(std::string json_file
                   , size_t with_batch_size )
    {
        std::string t;
        return this->init(json_file, with_batch_size, t);
    }

    bool RBM::init(std::string json_file
                   , size_t with_batch_size
                   , std::string& name)
    {
        std::ifstream inputFile(json_file);

        if (inputFile.good()) {
            this->json_path.assign(json_file);

            std::string json_source((std::istreambuf_iterator<char>(inputFile)),
                    std::istreambuf_iterator<char>());

            char json_str[json_source.length() + 1];
            strcpy(json_str, json_source.c_str());
            block_allocator alloc(1 << 10);

            json_value *json = parseJSON(json_str, &alloc);
            size_t last_path_split = json_file.find_last_of('/');
            std::string root_path;

            if (last_path_split == std::string::npos) {
                root_path = "./";
            } else {
                root_path = json_file.substr(0, last_path_split);
            }

            return this->init(json, root_path, with_batch_size, name, spLayer(NULL));

        } else {
            std::cerr << "Unable to open " << json_file << std::endl;
            exit(-1);
        }

        return false;
    }

    bool RBM::init(json_value* json
                   , std::string& directory
                   , size_t with_batch_size
                   , std::string& name
                   , spLayer visible)
    {
        this->rootdir.assign(directory);

        if (json->type == JSON_STRING) {
            std::string path_to_def = abspath_from_relative(json->string_value,
                    directory, false);

            return this->init(path_to_def, with_batch_size, name);

        } else {
            assert(json->type == JSON_OBJECT);

            this->init_set_params(json, with_batch_size);

            this->hidden = spLayer(new Layer(this->context));
            this->hidden->is_visible = false;
            if (visible == NULL) {
                this->visible = spLayer(new Layer(this->context));
            } else {
                this->visible = visible;
            }
            this->visible->is_visible = true;

            for (json_value *it = json->first_child; it; it = it->next_sibling) {
                if (it->name != NULL) {
                    if (strcmp(it->name, "hidden") == 0) {
                        this->hidden->init_blocks_json(it, &this->rng_states,
                                this->params.batchSize);
                    } else if ((strcmp(it->name, "visible") == 0) && visible == NULL) {
                        this->visible->init_blocks_json(it, &this->rng_states,
                                this->params.batchSize);
                    }
                }
            }

            this->init_core();

            if(name.length() > 0){ this->name.assign(name); }

            this->load(this->rootdir);

            if (!this->hidden->isInitialized()) {
                std::cerr << this->name << ": Unable to setup hidden layer" << std::endl;
                return false;
            }

            if (this->visible->data.cols() > 0) {
                this->init_with_visible(this->visible->data.cols());
            }

            return true;
        }
    }

    void RBM::init_set_params(json_value *root, size_t with_batch_size) {
        this->name.assign("RBM");
        this->epochs = 10;
        this->fixed_weights = false;
        this->useChurn = true;
        this->params.symWeights = true;

        for (json_value *it = root->first_child; it; it = it->next_sibling) {

            if (it->name != NULL) {
                if (strcmp(it->name, "name") == 0) {
                    this->name.assign(it->string_value);
                } else if (strcmp(it->name, "batch_size") == 0) {
                    this->params.batchSize = it->int_value;
                } else if (strcmp(it->name, "churn") == 0) {
                    this->useChurn = (it->int_value == 1);
                } else if (strcmp(it->name, "epsilon") == 0) {
                    this->params.epsilon = it->float_value;
                } else if (strcmp(it->name, "epochs") == 0) {
                    this->epochs = it->int_value;
                } else if (strcmp(it->name, "min_updates") == 0) {
                    this->minUpdates = it->int_value;
                } else if (strcmp(it->name, "max_updates") == 0) {
                    this->maxTrainingUpdates = it->int_value;
                } else if (strcmp(it->name, "effect_size") == 0) {
                    this->maxEffectSize = it->float_value;
                } else if (strcmp(it->name, "biasDecay") == 0) {
                    this->params.biasDecay = it->float_value;
                } else if (strcmp(it->name, "fixed_weights") == 0) {
                    this->fixed_weights = (it->int_value == 1);
                } else if (strcmp(it->name, "gibbs") == 0) {
                    this->params.gibbs = it->int_value;
                } else if (strcmp(it->name, "momentum") == 0) {
                    this->params.momentum = it->float_value;
                } else if (strcmp(it->name, "sparsityTarget") == 0) {
                    this->params.sparsityTarget = it->float_value;
                } else if (strcmp(it->name, "sparsityCost") == 0) {
                    this->params.sparsityCost = it->float_value;
                } else if (strcmp(it->name, "statLen") == 0) {
                    this->params.statLen = it->int_value;
                } else if (strcmp(it->name, "weightCost") == 0) {
                    this->params.weightCost = it->float_value;
                }
            }
        }

        if (with_batch_size > 0) {
            this->params.batchSize = with_batch_size;
        }

        this->params.epsilonDivBatch = this->params.epsilon / this->params.batchSize;

        this->using_momentum = (this->params.momentum != 0.0) && !this->fixed_weights;
    }

    bool RBM::init_core() {

        assert(this->params.batchSize > 0);
        if (this->context != nullptr) {
            this->kUpdateWeights = this->context->getKernel("update_weights");
            this->kWeightPenalty = this->context->getKernel("weight_penelty_term");
        }
        this->feed_forward();
        return true;
    }

    bool RBM::init_with_visible(size_t visible_width) {

        if (this->visible->isInitialized()
                && (this->visible->data.cols() != visible_width)) {
            std::cerr << "Conflicting visible sizes " << visible_width << " vs "
                    << this->visible->data.cols() << std::endl;
        }

        if (!this->visible->isInitialized()) {
            this->visible->init(visible_width, this->params.batchSize,
                    &this->rng_states);
        }

        size_t hidden_width = this->hidden->data.cols();
        size_t widest = std::max(visible_width, hidden_width);

        this->params.nvis = visible_width;
        this->params.nhid = hidden_width;

        if (this->rng_states.n_cells == 0) {
            this->rng_states.init({widest, 4});
        }

        srand(time(NULL));
        for (size_t x = 0; x < widest; x++) {
            this->rng_states.at(x, 0) = rand();
            this->rng_states.at(x, 1) = rand();
            this->rng_states.at(x, 2) = rand();
            this->rng_states.at(x, 3) = rand();
        }

        if (this->weights_vxh.n_cells == 0) {
            this->weights_vxh.init({visible_width, hidden_width});
            this->weights_hxv.init({hidden_width, visible_width});
        }

        if (this->using_momentum && (this->velocity_vxh.n_cells == 0)) {
            this->velocity_vxh.init({visible_width, hidden_width});
            this->velocity_hxv.init({hidden_width, visible_width});
        }

        if (this->kernelReturn.n_cells == 0) {
            this->kernelReturn.init({1, widest});
        }

        this->fillRand();

        if (this->params.sparsityTarget > 0) {
            decimal_t * hc = this->hidden->biasExp.getHost();
            for (size_t i = 0; i < this->nhid(); i++) {
                hc[i] = this->params.sparsityTarget;
            }
        }

        return true;
    }

    /**************
     * get_vxh should be true if requesting weights to generate the
     * hidden layer, false otherwise.
     */
    RBM::Matrix& RBM::weights(bool get_vxh) {
        if (get_vxh == this->feeding_forward)
            return this->weights_vxh;
        else
            return this->weights_hxv;
    }

    RBM::Matrix& RBM::velocity(bool get_vxh) {
        if (get_vxh == this->feeding_forward)
            return this->velocity_vxh;
        else
            return this->velocity_hxv;
    }

    void RBM::fillRand() {

        if (!this->_loaded_weights) {

            std::cerr << this->name << ": Filling with random values" << std::endl;

            this->visible->biases.set(0);
            this->hidden->biases.set(0); //.fillRandNormal(-1.8, 0.25);
            this->weights_hxv.fillRandNormal(0.0, 0.01);
            this->weights_hxv.transpose_to(this->weights_vxh);

            if (this->using_momentum) {
                this->velocity_hxv.set(0);
                this->velocity_vxh.set(0);
            }

        }

    }

    /** ------------------------------ Members ------------------------------ **/

    void RBM::feed_forward() {
        if (!this->feeding_forward) {
            std::swap(this->visible, this->hidden);
            this->feeding_forward = true;
            assert(this->weights(true).rows() == this->visible->data.cols());
            assert(this->weights(true).cols() == this->hidden->data.cols());
        }
    }

    void RBM::feed_backward() {
        if (this->feeding_forward) {
            std::swap(this->visible, this->hidden);
            this->feeding_forward = false;
            assert(this->weights(true).rows() == this->visible->data.cols());
            assert(this->weights(true).cols() == this->hidden->data.cols());
        }
    }

    void RBM::run(bool genHidden) {
        auto h = this->hidden;
        auto v = this->visible;

        if (genHidden) {
            h->gen_data(v->data, this->weights(true), this->params);
            h->sample(h->data, h->dataSample, this->params);
            v->gen_recon(h->dataSample, this->weights(false), this->params);
        } else {
            v->gen_recon(h->data, this->weights(false), this->params);
        }

        this->gibbs_sample(this->params.gibbs);
    }

    void RBM::run(Matrix& hiddenData) {
        this->visible->gen_recon(hiddenData, this->weights(false), this->params);
        this->gibbs_sample(this->params.gibbs);
    }

    void RBM::gibbs_sample(size_t gibbs) {
        for (size_t i = 1; i <= gibbs; i++) {
            this->hidden->gen_recon(this->visible->recon, this->weights(true),
                    this->params);
            this->visible->gen_recon(this->hidden->recon, this->weights(false),
                    this->params);
        }
    }

    /**
     * Train the layer using Contrastive Divergence
     **/

    void RBM::trainingLog(size_t batch, StatCounter& data_error,
            StatCounter& holdout_error) {

        decimal_t d_error = data_error.front();
        decimal_t h_error =
                (holdout_error.front() > 0) ?
                holdout_error.front() : std::numeric_limits<decimal_t>::quiet_NaN();

        this->trainingCLILog(batch, data_error, holdout_error);

        this->visible->set_error();

        this->logError(d_error, h_error, false);
        this->logHistograms(this->output_options.hist_bin_count);
        this->logFreeEnergy();
        this->logEffect();
    }

    void RBM::trainingCLILog(size_t batch, StatCounter& data_error,
            StatCounter& holdout_error) {

        if (this->output_options.update_cli) 
        {
            std::clog << std::left << std::setw(5) << "\rE " << this->epoch_counter << " ("<< this->batch_counter <<")" << ": ";
    
            std::clog << "Error: " << std::left << std::fixed << std::setprecision(6)
                    << data_error.mean() << " +/- " << data_error.stdev();
    
            if (this->params.weightCost > 0) {
                std::clog << " Decay: " << this->params.decay;
            }
    
            if( this->dataEntropy.nObs() > 0 ){
                std::clog << " Effect: " << std::left << std::fixed << std::setprecision(6) << this->getEffectSize();
            }
    
            if (holdout_error.nObs() > 0) {
                std::clog << " Holdout: " << std::left << std::fixed << std::setprecision(6)
                        << holdout_error.mean() << " +/- " << holdout_error.stdev();
            }
    
            std::clog << std::flush;
        }
    }

    void RBM::train(RBM::Spout dataSpout, RBM::Spout holdoutSpout) {
        
        if( this->output_options.update_cli ){
            std::cerr << "Updating CLI" << std::endl;
        }else{
            std::cerr << "Not updating CLI" << std::endl;
        }
        
        if (!this->fixed_weights) {
            std::vector<size_t> dataShape = { this->visible->data.rows(),
                                              this->visible->data.cols() };

            RBM::Matrix dataChurn(NULL);
            decimal_t batch_error;
            size_t logCheck = this->output_options.log_per_n_batches;

            this->logSetup(this->epochs);

            if (this->useChurn) {
                dataChurn.init(dataShape);
                if (!dataSpout(dataChurn)) {
                    std::clog
                            << "You're training over one batch. This probably will not end well..."
                            << std::endl;
                }
            }

            this->batch_counter=0;
            this->epoch_counter=0;

            std::clog << "Training" << "\n\tVisible: " << this->visible->data.cols()
                    << " units" << "\n\tHidden: " << this->hidden->data.cols()
                    << " units" << "\n\t\t(built " << __TIME__ << ")" << std::endl;

            while ( this->shouldKeepTraining() ) {

                StatCounter dataErrorCounter;
                StatCounter holdoutErrorCounter;

                this->logEpoch(this->epoch_counter, this->epochs);

                while ( dataSpout(this->visible->data) && this->shouldKeepTraining() ) {

                    if (this->useChurn) {
                        dataChurn.churn(this->visible->data);
                    }

                    this->batch_counter++;

                    this->run(true);
                    this->update();
                    batch_error = this->error();
                    dataErrorCounter.push(batch_error);

                    if (holdoutSpout != NULL && holdoutSpout(this->visible->data)) {
                        this->run(true);
                        holdoutErrorCounter.push(this->error());
                    }

                    if ((logCheck > 0) && (this->batch_counter % logCheck) == 0) {

                        this->trainingLog(this->batch_counter
                                          , dataErrorCounter
                                          , holdoutErrorCounter);

                    } else if (logCheck <= 0
                                && this->output_options.update_cli
                                && ((this->batch_counter % 13) == 0)) {

                        this->trainingCLILog(this->batch_counter
                                             , dataErrorCounter
                                             , holdoutErrorCounter);

                    }

                }

                this->trainingCLILog(this->batch_counter, dataErrorCounter, holdoutErrorCounter);

                if ((logCheck > 0) && dataErrorCounter.nObs() > 0) {
                    this->logMeanError(dataErrorCounter, holdoutErrorCounter);
                }

                this->epoch_counter++;
            }
        }
    }

    void RBM::cross_train(Spout dataSpout, Spout featureSpout,
            Spout dataHoldoutSpout, Spout featureHoldoutSpout) {
        if (!this->fixed_weights) {
            float rnum;
            Matrix dataChurn(NULL);
            Matrix featuresChurn(NULL);

            size_t logCheck = this->output_options.log_per_n_batches;
            decimal_t batch_error;

            this->logSetup(this->epochs);

            if (this->useChurn) {
                dataSpout(dataChurn);
                featureSpout(featuresChurn);
            }

            while( this->shouldKeepTraining() ) {
                std::clog << "Epoch: " << this->epoch_counter << std::endl;

                StatCounter dataError;
                StatCounter holdoutError;

                this->logEpoch(this->epoch_counter, this->epochs);

                while (dataSpout(this->visible->data) && featureSpout(this->hidden->data) && this->shouldKeepTraining()) {
                    if (this->useChurn) {
                        for (size_t r = 0; r < dataChurn.rows(); r++) {
                            rnum = (float) rand() / (float) RAND_MAX;
                            if (rnum > 0.2) {
                                dataChurn.swap_row(this->visible->data, r, r);
                                featuresChurn.swap_row(this->hidden->data, r, r);
                            }
                        }
                    }

                    this->batch_counter++;
                    this->run(false);
                    this->update();

                    batch_error = this->error();
                    dataError.push(batch_error);

                    if (dataHoldoutSpout != NULL && featureHoldoutSpout != NULL) {
                        // Wrap this to prevent a holdout epoch
                        // from triggering training epoch
                        if (dataHoldoutSpout(this->visible->data)
                                && featureHoldoutSpout(this->hidden->data)) {
                            this->run(false);
                            holdoutError.push(this->error());
                        }

                    }

                    if ((logCheck > 0) && (this->batch_counter % logCheck == 0)) {
                        this->trainingLog(this->batch_counter, dataError, holdoutError);
                    } else if (logCheck <= 0 && this->output_options.update_cli
                            && (this->batch_counter % 13 == 0)) {
                        this->trainingCLILog(this->batch_counter, dataError, holdoutError);
                    }
                } // End of the while

                std::clog << std::endl;

                if ((logCheck > 0) && dataError.nObs() > 0) {
                    this->logMeanError(dataError, holdoutError);
                }
            }
        }
    }

    void RBM::cross_train(RBM& crossModel, Spout myData, Spout crossData) {

        if (!this->fixed_weights) {
            float rnum;
            Matrix myChurn(NULL);
            Matrix crossChurn(NULL);

            size_t logCheck = this->output_options.log_per_n_batches;
            decimal_t batch_error;

            this->logSetup(this->epochs);

            if (this->useChurn) {
                myChurn.init(this->visible->data);
                crossChurn.init(crossModel.visible->data);
                myData(myChurn);
                crossData(crossChurn);
            }

            while ( this->shouldKeepTraining() ) {

                StatCounter dataError;
                StatCounter holdoutError;

                std::clog << "Epoch: " << this->epoch_counter << std::endl;

                this->logEpoch(this->epoch_counter, this->epochs);
                this->epoch_counter++;

                while (myData(this->visible->data) && crossData(crossModel.visible->data) && this->shouldKeepTraining() ) {

                    if (this->useChurn) {
                        for (size_t r = 0; r < myChurn.rows(); r++) {
                            rnum = (float) rand() / (float) RAND_MAX;
                            if (rnum > 0.2) {
                                myChurn.swap_row(this->visible->data, r, r);
                                crossChurn.swap_row(crossModel.visible->data, r, r);
                            }
                        }
                    }

                    this->batch_counter++;
                    crossModel.run(true);
                    this->run(crossModel.hidden->recon);
                    this->update();

                    batch_error = this->error();
                    dataError.push(batch_error);

                    if ((logCheck > 0) && (this->batch_counter % logCheck == 0)) {

                        this->trainingLog(this->batch_counter
                                          , dataError
                                          , holdoutError);

                    } else if (logCheck <= 0 && this->output_options.update_cli
                            && (this->batch_counter % 13 == 0)) {

                        this->trainingCLILog(this->batch_counter
                                             , dataError
                                             , holdoutError);

                    }
                }

                std::clog << std::endl;

                if ((logCheck > 0) && dataError.nObs() > 0) {
                    this->logMeanError(dataError, holdoutError);
                }
            }
        }
    }

    void RBM::reconstruct(zbin::SelectBox& section, size_t iterations,
            Spout dataSpout, Drain dataSink) {

        if (this->output_options.update_cli) {
            std::cout << "Reconstructing...\n";
        }

        if (section.endCol > this->nvis()) {
            section.endCol = this->nvis();
        }

        Matrix mask(this->context);
        OCLMatrix<cl_uint> maskSum(this->context);
        mask.init(this->visible->data);
        maskSum.init({1});

        size_t startRow = 0;
        size_t endRow = this->visible->data.rows();

        auto v = this->visible;

        while (dataSpout(v->data)) {

            if (endRow >= section.startRow && startRow < section.endRow) {

                size_t passNumber = 0;

                do {

                    this->run(true);

                    v->gradient_reconstruct(startRow, section, this->params);

                    if (this->output_options.update_cli) {
                        std::cout << "\r\tbatch " << std::right << std::setw(10)
                                << std::setfill(' ') << this->batch_counter << " pass "
                                << std::flush;
                    }

                } while (passNumber++ < iterations);

                dataSink(v->data);
            }

            this->batch_counter++;
            startRow = endRow;
            endRow += v->data.rows();
        }

        if (this->output_options.update_cli) {
            std::cout << std::endl;
        }

    }

    void RBM::gen_from_features(Spout featureSpout, Drain dataSink) {

        if (this->output_options.update_cli) {
            std::cout << "Generating from features...\n";
        }

        while (featureSpout(this->hidden->data)) {

            this->run(false);
            dataSink(this->visible->recon);

            if (this->output_options.update_cli) {
                std::cout << "\r\tbatch " << std::right << std::setw(10)
                        << this->batch_counter++ << std::flush;
            }

        }

        if (this->output_options.update_cli) {
            std::cout << std::endl;
        }

    }

    void RBM::gen_features(Spout dataSpout, Drain featureSink) {

        if (this->output_options.update_cli) {
            std::cout << "Generating features...\n";
        }

        while (dataSpout(this->visible->data)) {

            this->hidden->gen_data(this->visible->data, this->weights(true),
                    this->params);

            if (this->output_options.update_cli) {
                std::cout << "\r\tbatch " << std::right << std::setw(10)
                        << this->batch_counter++ << std::flush;
            }

            featureSink(this->hidden->data);
        }

        if (this->output_options.update_cli) {
            std::cout << std::endl;
        }
    }

    void RBM::cross_gen(size_t iters, zbin::SelectBox& section, RBM& crossModel,
            Spout dataInput, Drain dataOutput) {

        if (this->output_options.update_cli) {
            std::cout << "Cross building...\n";
        }

        if (section.endCol > crossModel.nvis()) {
            section.endCol = crossModel.nvis();
        }

        assert(this->params.batchSize == crossModel.params.batchSize);

        size_t startRow = 0;
        size_t endRow = this->params.batchSize;

        while (dataInput(this->visible->data)) {

            if (this->output_options.update_cli) {
                std::cout << "\r\tbatch " << std::right << std::setw(10)
                        << this->batch_counter++ << std::flush;
            }

            this->hidden->gen_data(this->visible->data, this->weights(true),
                    this->params);

            crossModel.visible->gen_data(this->hidden->data, crossModel.weights(false),
                    this->params);

            for (size_t i = 0; i < iters; i++) {
                crossModel.run(true);
                crossModel.visible->gradient_reconstruct(startRow, section,
                        crossModel.params);
            }

            startRow = endRow;
            endRow += this->params.batchSize;

            dataOutput(crossModel.visible->data);
        }

        if (this->output_options.update_cli) {
            std::cout << std::endl;
        }
    }

    void RBM::calc_energy(Spout dataSpout, Drain energySink) {

        if (this->output_options.update_cli) {
            std::cout << "Calculating energy...\n";
        }

        while ( dataSpout(this->visible->data) ) {

            this->visible->set_energy(true, this->hidden, this->weights(true),
                    this->params);

            if (this->output_options.update_cli) {
                std::cout << "\r\tbatch " << std::right << std::setw(10)
                        << this->batch_counter++ << std::flush;
            }

            energySink(this->visible->energy);
        }

        if (this->output_options.update_cli) {
            std::cout << std::endl;
        }
    }

    void RBM::print_col_error(Spout input, Drain output) {

        StatCounter counters[ this->nvis() ];

        while (input(this->visible->data)) {
            this->run(true);
            this->visible->set_column_error(counters);
        }

        std::clog << "column,low,high,mean,stdev\n";
        for (size_t i = 0; i < this->nvis(); i++) {
            StatCounter& c = counters[i];
            std::clog << i << "," << c.min()
                    << "," << c.max()
                    << "," << c.mean()
                    << "," << c.stdev() << "\n";
        }
        std::clog << std::flush;
    }

    void RBM::print_row_error(Spout input, Drain output) {

        ailab::StatCounter counters[ this->params.batchSize ];

        size_t r = 0;
        std::clog << "row,low,high,mean,stdev\n";

        while (input(this->visible->data)) {
            this->run(true);
            this->visible->set_row_error(counters);

            for (size_t i = 0; i < this->params.batchSize; i++) {
                StatCounter& c = counters[i];
                std::clog << r++
                        << "," << c.min()
                        << "," << c.max()
                        << "," << c.mean()
                        << "," << c.stdev() << "\n";
            }
        }

        std::cout << std::flush;

    }

    void RBM::update() {

        if (this->params.weightCost > 0 && this->prevWeightPenalty == 0) {
            this->prevWeightPenalty = this->get_weight_penalty()
                    * this->params.weightCost;
        }

        this->wasUpdated = true;
        this->n_updates++;
        this->visible->update_bias(this->params);
        this->hidden->update_bias(this->params);
        this->update_weights();

        if (this->params.weightCost > 0) {
            decimal_t newPenalty = this->get_weight_penalty() * this->params.weightCost;

            if (this->prevWeightPenalty) {
                this->params.decay = this->params.epsilon
                        * (this->prevWeightPenalty - newPenalty);
            } else {
                this->params.decay = 0;
            }

            assert(std::isfinite(this->params.decay));
            this->prevWeightPenalty = newPenalty;
        }
    }

    decimal_t RBM::get_weight_penalty() {

        this->kernelReturn.set(0);
        decimal_t penalty = 0.0;

        if (this->kWeightPenalty != NULL) {

            assert(this->kernelReturn.getCL() != NULL);
            assert(this->weights(true).getCLConst());

            this->kernelReturn.getHost();
            this->kernelReturn.set(0);

            this->kWeightPenalty->set_size(this->nhid());
            this->kWeightPenalty->run(this->kernelReturn.getCL(),
                    this->params,
                    this->weights(true).getCLConst());

            this->kernelReturn.getHost();
            for (size_t j = 0; j < this->nhid(); j++) {
                penalty += this->kernelReturn.at(j);
            }

        } else {

            Matrix& w = this->weights(true);
            for (size_t j = 0; j < this->nhid(); j++) {
                for (size_t i = 0; i < this->nvis(); i++) {
                    penalty += std::pow(w.at(i, j), 2);
                }
            }

        }

        if (!std::isfinite(penalty)) {
            std::cerr << "\n" << this->name
                    << ": This RBM has destabilized, try different parameters"
                    << std::endl;
            exit(-1);
        }

        return penalty / 2;
    }

    void RBM::update_weights() {
        OpenCL::spKernel k = this->kUpdateWeights;

        if (k != NULL) {
            k->set_size(this->nvis(), this->nhid());
            k->run(this->weights(true).getCL(),
                    this->velocity(true).getCL(),
                    this->weights(false).getCL(),
                    this->velocity(false).getCL(),
                    this->visible->data.getCLConst(),
                    this->visible->recon.getCLConst(),
                    this->hidden->data.getCLConst(),
                    this->hidden->recon.getCLConst(),
                    this->params);
        } else {

            RBMParams& p = this->params;
            Matrix& w = this->weights(true);
            Matrix& v = this->velocity(true);
            Matrix& wT = this->weights(false);
            Matrix& vT = this->velocity(false);

            Matrix& vData = this->visible->data;
            Matrix& vRecon = this->visible->recon;

            Matrix& hData = this->hidden->data;
            Matrix& hRecon = this->hidden->recon;

            decimal_t exp_sisj_data = 0.0f;
            decimal_t exp_sisj_model = 0.0f;
            decimal_t w_ij = 0;
            decimal_t v_ij = 0;
            decimal_t delta_ij = 0;
            size_t nvis = this->nvis();
            size_t nhid = this->nhid();
            size_t i, j, r;

            assert(w.rows() == nvis);
            assert(w.cols() == nhid);

            if (this->params.symWeights) {
                wT.getHost();
                vT.getHost();
            }

            w.getHost();
            v.getHost();
            vData.getHostConst();
            vRecon.getHostConst();
            hData.getHostConst();
            hRecon.getHostConst();

            for (i = 0; i < nvis; i++) {
                for (j = 0; j < nhid; j++) {
                    w_ij = w.at(i, j);

                    exp_sisj_data = 0.0f;
                    exp_sisj_model = 0.0f;

                    for (r = 0; r < p.batchSize; r++) {
                        exp_sisj_data += vData.at(r, i) * hData.at(r, j);
                        exp_sisj_model += vRecon.at(r, i) * hRecon.at(r, j);
                    }

                    delta_ij = (exp_sisj_data - exp_sisj_model);
                    delta_ij = p.epsilonDivBatch * (delta_ij - std::copysign(p.decay, delta_ij));

                    if (p.momentum > 0) {
                        v_ij = (p.momentum * v.at(i, j)) + delta_ij;
                        delta_ij = v_ij;
                        v.at(i, j) = v_ij;
                    }

                    w_ij += delta_ij;
                    ;

                    w.at(i, j) = w_ij;
                    if (this->params.symWeights) {
                        wT.at(j, i) = w_ij;
                        if (p.momentum > 0) {
                            vT.at(j, i) = v_ij;
                        }
                    }
                }
            }
        }
    }

    bool RBM::weights_are_symmetric() {
        return this->params.symWeights;
    }

    void RBM::untie_weights() {
        this->params.symWeights = false;
    }

    size_t RBM::nvis() {
        return this->visible->data.cols();
    }

    size_t RBM::nhid() {
        return this->hidden->data.cols();
    }

    decimal_t RBM::error() {
        double mae = this->visible->data.mae(this->visible->recon);
        auto& s = this->errorStats;

        s.count++;
        if (s.count == 1) {
            s.oldMean = s.newMean = mae;
            s.oldSum = 0;
        } else {
            s.newMean = s.oldMean + (mae - s.oldMean) / s.count;
            s.newSum = s.oldSum + (mae - s.oldMean) * (mae - s.newMean);
            s.oldMean = s.newMean;
            s.oldSum = s.newSum;
        }

        return mae;
    }

    decimal_t RBM::errStdev() {
        auto& s = this->errorStats;
        return (s.count > 1) ? std::sqrt(s.newSum / (s.count - 1)) : 0.0;
    }

    decimal_t RBM::errMean() {
        return this->errorStats.newMean;
    }

    RBM::Matrix& RBM::get_visible_energy(bool forData) {
        this->visible->set_energy(forData, this->hidden, this->weights(true),
                this->params);
        return this->visible->energy;
    }

    /**
     * @brief RBM::getEffectSize
     * @return The effect size, using a maximum likelihood estimator.
     * ( Larry V. Hedges & Ingram Olkin (1985). Statistical Methods for Meta-Analysis. Orlando: Academic Press. ISBN 0-12-336380-2.)
     */
    decimal_t RBM::getEffectSize(){
        StatCounter& d = this->dataEntropy;
        StatCounter& r = this->reconEntropy;
        return (r.mean() - d.mean()) / d.stdev();
    }

    bool RBM::verify_ready() {
        auto h = this->hidden;
        auto v = this->visible;

        bool ready = true;

        if (!(h->isInitialized())) {
            std::cerr << "Hidden layer not initialized" << std::endl;
            ready = false;
        } else if (!(h->data.cols() > 0)) {
            std::cerr << "Hidden layer has 0 width" << std::endl;
            ready = false;
        }

        if (!(v->isInitialized())) {
            std::cerr << "Visible layer not initialized" << std::endl;
            ready = false;
        } else if (!(v->data.cols() > 0)) {
            std::cerr << "Visible layer has 0 width" << std::endl;
            ready = false;
        }

        if (!(v->data.rows() == this->params.batchSize)) {
            std::cerr << "Visible batch size mismatched" << std::endl;
            ready = false;
        }

        if (!(h->data.rows() == this->params.batchSize)) {
            std::cerr << "Hidden batch size mismatched" << std::endl;
            ready = false;
        }

        if (!(this->weights_vxh.rows() == v->data.cols())) {
            std::cerr << "Visible size doesn't match weights" << std::endl;
            ready = false;
        }

        if (!(this->weights_vxh.cols() == h->data.cols())) {
            std::cerr << "Hidden size doesn't match weights" << std::endl;
            ready = false;
        }

        if (!(std::equal(v->data.dims.begin(), v->data.dims.end(),
                v->recon.dims.begin()))) {
            std::cerr << "Visible data doesn't match visible recon" << std::endl;
            ready = false;
        }

        if (!(std::equal(h->data.dims.begin(), h->data.dims.end(),
                h->recon.dims.begin()))) {
            std::cerr << "Hidden data doesn't match hidden recon" << std::endl;
            ready = false;
        }

        /**
         * Do this to verify that the buffers can read/write
         */
        if (ready) {
            ready &= this->visible->biases.check(true);
            ready &= this->visible->data.check(true);
            ready &= this->visible->recon.check(true);
            ready &= this->hidden->data.check(true);
            ready &= this->hidden->recon.check(true);
            ready &= this->hidden->biases.check(true);
            ready &= this->weights_hxv.check(true);
            ready &= this->weights_vxh.check(true);

            if (this->using_momentum) {
                ready &= this->velocity_hxv.check(true);
                ready &= this->velocity_vxh.check(true);
            }

        }

        return ready;
    }

    bool RBM::shouldKeepTraining(){

        if(this->minUpdates > this->batch_counter){
            return true;
        }else if(this->epochs > 0 && this->epoch_counter >= this->epochs){
            return false;
        }else if(this->maxTrainingUpdates > 0 && this->batch_counter >= this->maxTrainingUpdates){
            return false;
        }else{

            LayerDiff diff = this->visible->get_entropy();
            this->dataEntropy.push(diff.data);
            this->reconEntropy.push(diff.recon);

            if((this->output_options.log_per_n_batches > 0) && ((this->batch_counter % this->output_options.log_per_n_batches) == 0) )
            {
                this->logEntropy(diff);
            }

            /* Wait until we have a large enough population to
             * estimate entropy differences.
             */
            if( (this->maxEffectSize > 0)
                    && (this->dataEntropy.nObs() > 30)
                    && (this->getEffectSize() < this->maxEffectSize) ){
                return false;
            }
        }

        return true;
    }

    /*-----------------------------------------------
     Logging Methods
     ----------------------------------------------*/
    void RBM::logSetup(unsigned int epochs) {

        std::stringstream json;
        json << ",\"setup\":\"" << this->json_path << "\"";
        json << ",\"with_momentum\":" << (this->using_momentum ? "true" : "false");
        json << ",\"nvis\":" << this->nvis();
        json << ",\"nhid\":" << this->nhid();
        json << ",\"nbins\":" << this->output_options.hist_bin_count;
        json << ",\"epochs\":" << epochs;

        json << ",\"params\" : {\"epsilon\":" << this->params.epsilon
                << ",\"epsilonDivBatch\":" << this->params.epsilonDivBatch
                << ",\"decay\":" << this->params.decay
                << ",\"biasDecay\":" << this->params.biasDecay
                << ",\"sparsityTarget\":" << this->params.sparsityTarget
                << ",\"sparsityCost\":" << this->params.sparsityCost
                << ",\"weightCost\":" << this->params.weightCost
                << ",\"momentum\":" << this->params.momentum
                << ",\"gibbs\":" << this->params.gibbs
                << ",\"batchSize\":" << this->params.batchSize
                << ",\"statLen\":" << this->params.statLen
                << ",\"symWeights\":" << (this->params.symWeights ? "true" : "false")
                << ",\"nvis\":" << this->params.nvis
                << ",\"nhid\":" << this->params.nhid;
        json << "}";

        this->log(json);
    }

    void RBM::logEpoch(unsigned int e, unsigned int epoch_count) {

        std::stringstream json;
        json << ",\"epoch\":" << e;
        this->log(json);
    }

    void RBM::logHistograms(size_t nbins, bool backprop) {
        if (this->output_options.logHistograms) {
            std::stringstream json;
            std::string stdJson = backprop ? ",\"bp\":true" : "";

            json << ",\"histograms\":true";

            // Visible
            json << ",\"visible\":";
            this->visible->data.statsToJson(json, nbins);

            // Visible reconstruction
            json << ",\"visible_recon\":";
            this->visible->recon.statsToJson(json, nbins);

            // Hidden, always sampled. Just count 1s
            size_t hidden_cells = this->hidden->data.n_cells;
            double hidden_data_on = 0;
            double hidden_recon_on = 0;
            const decimal_t* hdata = this->hidden->data.getHostConst();
            const decimal_t* hrecon = this->hidden->recon.getHostConst();

            for (size_t i = 0; i < hidden_cells; i++) {
                hidden_data_on += hdata[i];
                hidden_recon_on += hrecon[i];
            }

            // Hidden counts
            json << ",\"hidden\":" << (hidden_data_on / hidden_cells);
            json << ",\"hidden_recon\":" << (hidden_recon_on / hidden_cells);

            // Visible Biases
            json << ",\"visible_biases\":";
            this->visible->biases.statsToJson(json, nbins);

            // Hidden Biases
            json << ",\"hidden_biases\":";
            this->hidden->biases.statsToJson(json, nbins);

            // Weights
            json << ",\"weights\":";
            this->weights_vxh.statsToJson(json, nbins);

            if (!this->params.symWeights) {
                json << ",\"weights_up\":";
                this->weights_hxv.statsToJson(json, nbins);
            }

            // Momentum ?
            if (this->using_momentum) {
                json << ",\"momentum\":";
                this->velocity_vxh.statsToJson(json, nbins);

                if (!this->params.symWeights) {
                    json << ",\"momentum_up\":";
                    this->velocity_hxv.statsToJson(json, nbins);
                }
            }

            this->log(json);
        }
    }

    void RBM::logFreeEnergy() {
        if (this->output_options.logEnergy) {
            std::stringstream json;
            this->visible->set_energy(true, this->hidden, this->weights(true),
                    this->params);
            Matrix& e = this->visible->energy;
            e.getHostConst();
            json << ",\"energy\":[";
            json << e.at(0);
            for (size_t i = 1; i < this->params.batchSize; i++) {
                json << "," << e.at(i);
            }
            json << "]";
            this->log(json);
        }
    }

    void RBM::logError(decimal_t d, decimal_t h, bool backprop) {
        if (this->output_options.logError) {
            std::stringstream json;

            if (backprop)
                json << ",\"bp\":true";

            json << ",\"error\": { \"data\" : " << d;
            if (h == h)
                json << ",\"holdout\":" << h;

            if (this->output_options.logErrorDetails) {

                this->visible->set_error();

                size_t i = 0;
                size_t nRows = this->visible->row_error.n_cells;
                size_t nCols = this->visible->col_error.n_cells;
                const decimal_t* R = this->visible->row_error.getHostConst();
                const decimal_t* C = this->visible->col_error.getHostConst();

                if (nRows > 0) {
                    json << ",\"row\":[";
                    json << R[0];
                    for (i = 0; i < nRows; i++)
                        json << "," << R[i];
                    json << "]";
                }

                if (nCols > 0) {
                    json << ",\"column\":[";
                    json << C[0];
                    for (i = 0; i < nCols; i++)
                        json << "," << C[i];
                    json << "]";
                }
            }

            json << "}";
            this->log(json);
        }
    }

    void RBM::logMeanError(StatCounter& data_error, StatCounter& holdout_error,
            bool backprop) {
        if (this->output_options.logError && data_error.nObs() > 0) {
            std::stringstream json;

            if (backprop)
                json << ",\"bp\":true";

            json << ",\"epochError\": { \"data\": " << "{\"min\":" << data_error.min()
                    << ",\"max\":" << data_error.max() << ",\"mean\":" << data_error.mean()
                    << ",\"stdev\":" << data_error.stdev() << "} ";

            if (holdout_error.nObs() > 0) {
                json << ",\"holdout\":" << "{\"min\":" << holdout_error.min()
                        << ",\"max\":" << holdout_error.max() << ",\"mean\":"
                        << holdout_error.mean() << ",\"stdev\":" << holdout_error.stdev()
                        << "} ";
            }

            json << "}";

            this->log(json);
        }
    }


    void RBM::logEffect() {
        if(this->output_options.logEffect){
            decimal_t x = this->getEffectSize();
            if(std::isfinite(x)){
                std::stringstream json;
                json << ",\"effect\":" << x;
                this->log(json);
            }
        }
    }

    void RBM::logEntropy(LayerDiff &diff){
        if( this->output_options.logEntropy ){
            std::stringstream json;
            json << ",\"data_entropy\":" << diff.data << ",\"recon_entropy\":" << diff.recon;
            this->log(json);
        }
    }

    /**
     * Log wraps incoming keys in an object for JSON
     * @brief RBM::log
     * @param doc
     */
    void RBM::log(std::stringstream& io) {
        io << ",\"n\":" << this->n_updates;
        io << ",\"batch\":" << this->batch_counter;
        io << ",\"rbm\":\"" << this->name << "\"";
        Logger::log("jlog", io);
    }

    bool RBM::save() {

        bool write_both_matrices = !this->weights_are_symmetric();
        this->weights_vxh.getHostConst();
        this->weights_hxv.getHostConst();

        if (this->wasUpdated) {
            std::string directory = this->rootdir;
            directory.append("/");
            directory.append(this->name);
            directory.append(".weights");

            std::ofstream f(directory.c_str(),
                    std::ios_base::out | std::ios_base::binary);

            if ( f.good() ) {
                if (this->output_options.update_cli)
                    std::cout << "\n" << this->name << ": Saving state to " << directory
                        << " ... ";

                WeightsHeader header = {
                    RBM_STRUCT_VERSION
                    ,sizeof(decimal_t)
                    ,write_both_matrices ? 1 : 0
                    ,this->using_momentum ? 1 : 0
                    ,this->n_updates
                    ,this->nvis()
                    ,this->nhid()
                };

                f.write((char *) &header, sizeof (header));

                this->visible->biases.write_to(f);
                this->hidden->biases.write_to(f);

                this->weights_vxh.write_to(f);
                if ( write_both_matrices ) {
                    this->weights_hxv.write_to(f);
                }

                if (this->using_momentum) {
                    this->velocity_vxh.write_to(f);
                    if ( write_both_matrices ) {
                        this->velocity_hxv.write_to(f);
                    }
                }

                f.close();

                if (this->output_options.update_cli)
                {
                    std::cout << "done!" << std::endl;
                }

                return true;
            }

        }
        return false;
    }

    bool RBM::isSetup() {

        RBM::spLayer v = this->visible;
        RBM::spLayer h = this->hidden;

        return this->visible->isInitialized() && this->hidden->isInitialized()
                && v->data.n_cells > 0 && v->data.n_cells == v->recon.n_cells
                && h->data.n_cells > 0 && h->data.n_cells == h->recon.n_cells;
    }

    bool RBM::load(std::string directory) {
        directory.append("/");
        directory.append(this->name);
        directory.append(".weights");
        
        std::ifstream f(directory.c_str(), std::ios_base::in | std::ios_base::binary);

        if (f.good()) {
            WeightsHeader header;

            f.read((char *) &header, sizeof(header));

            if(header.version != RBM_STRUCT_VERSION){
                std::cerr << this->name << ": Could not load weights, wrong version. In header "
                        << header.version << ", my version is " << RBM_STRUCT_VERSION << std::endl;
                f.close();
                return false;
            }

            if ( header.weight_bytes != (sizeof (decimal_t)) ) {
                std::cerr << this->name << ": Type size mismatch decimal_t should be "
                        << (sizeof (decimal_t)) << " not " << header.weight_bytes << std::endl;
                f.close();
                return false;
            }

            if (((this->nvis() != 0) && (header.nvis != this->nvis()))
                    || ((this->nhid() != 0) && (header.nhid != this->nhid()))) {
                std::cerr << this->name << ": Shape mismatch, I am " << this->nvis()
                        << "x" << this->nhid();
                std::cerr << " the weights provided are " << header.nvis << "x" << header.nhid
                        << std::endl;
                f.close();
                return false;
            }

            this->n_updates = header.n_updates;
            this->batch_counter = this->n_updates;

            if (!this->hidden->isInitialized()) {
                this->hidden->init(header.nhid, this->params.batchSize, &this->rng_states);
            }

            if (!this->visible->isInitialized()) {
                this->init_with_visible(header.nvis);
            } else if (this->weights_vxh.n_cells == 0) {
                this->weights_vxh.init({header.nvis, header.nhid});
                this->weights_hxv.init({header.nhid, header.nvis});

                this->velocity_vxh.init({header.nvis, header.nhid});
                this->velocity_hxv.init({header.nhid, header.nvis});
            }

            this->visible->biases.read_from(f);
            this->hidden->biases.read_from(f);
            this->weights_vxh.read_from(f);

            if (header.dual_matrices) {
                //this->params.symWeights = false;
                this->weights_hxv.read_from(f);
            } else {
                //this->params.symWeights = true;
                this->weights_vxh.transpose_to(this->weights_hxv);
            }

            /**
             * Handling momentum
             */
            if (header.has_momentum == this->using_momentum) {
                if (header.has_momentum) {
                    this->velocity_vxh.read_from(f);
                    if (header.dual_matrices) {
                        this->velocity_hxv.read_from(f);
                    } else {
                        this->velocity_vxh.transpose_to(this->velocity_hxv);
                    }
                }

            } else if (this->using_momentum) {
                std::cerr
                        << this->name
                        << ": Missing momentum matrices. Initializing momentum to random..."
                        << std::endl;
                this->velocity_hxv.set(0);
                this->velocity_hxv.transpose_to(this->velocity_vxh);
            } else if (header.has_momentum) {
                std::cerr << this->name << ": Available momentum being discarded..."
                        << std::endl;
            }

            std::clog << this->name << ": Loaded saved state" << std::endl;
            this->_loaded_weights = true;
            return true;
        } else {
            return false;
        }

    }

}
