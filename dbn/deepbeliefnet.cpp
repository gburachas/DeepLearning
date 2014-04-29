#include <ailab/opencl.h>
#include <ailab/algorithms/rbm.h>
#include <ailab/algorithms/deepbeliefnet.h>
#include <ailab/statcounter.h>

#include <algorithm>
#include <chrono>
#include <iostream>
#include <fstream>
#include <functional>
#include <list>
#include <map>
#include <stdint.h>
#include <string>
#include <set>
#include <vector>
#include <queue>

#include <iomanip>

#include <libgen.h>

namespace ailab {

/**********************************************************
 *
 * DBN Node
 *
 *********************************************************/

DeepBeliefNet::Node::Node(DeepBeliefNet *net)
    : rbm(net->context),
      dbn(net) {
  this->rbm.output_options = net->output_options;
}

void DeepBeliefNet::Node::run_toward_root() {

  if (this->tree->children.size() > 0) {
    for (Tree<spNode>::spTree child : this->tree->children) {
      spNode node = child->value();
      node->run_toward_root();
      RBM::Matrix& childHidden = node->rbm.hidden->recon;
      this->rbm.visible->data.read_rect(childHidden, 0, 0, 0,
                                        node->parent_column_offset,
                                        childHidden.rows(), childHidden.cols());
    }

    this->rbm.run(true);
  } else {
    try {
      this->input(this->rbm.visible->data);
    } catch (Exception::IOReachedEnd& e) {
      this->dbn->input_is_exhausted();
    }
    this->rbm.run( true);
  }
}

void DeepBeliefNet::Node::run_toward_leaves() {
  this->rbm.run(false);
  if (this->tree->children.size() > 0) {
    for (Tree<spNode>::spTree child : this->tree->children) {
      spNode node = child->value();
      RBM::Matrix& childData = node->rbm.hidden->data;
      childData.read_rect(this->rbm.visible->recon, 0,
                          node->parent_column_offset, 0, 0, childData.rows(),
                          childData.cols());

      node->run_toward_leaves();
    }
  }
}

bool DeepBeliefNet::Node::isLeaf() {
  assert(this->tree != NULL);
  return this->tree->children.size() == 0;
}

bool DeepBeliefNet::Node::isRoot() {
  assert(this->tree != NULL);
  return this->tree->parent == NULL;
}

/**********************************************************
 *
 *  Deep Belief Networks implementation
 *
 *********************************************************/

DeepBeliefNet::DeepBeliefNet(OpenCL::spContext context)
    : CLBacked(context) {
  this->up_down_epochs = 10;
}

void DeepBeliefNet::init(std::string json_path, DataIO<ailab::decimal_t> &io,
                         size_t batch_size) {
  size_t pos = 0;
  std::ifstream inputFile(json_path);

  if (inputFile.is_open()) {
    this->raw_json.assign(std::istreambuf_iterator<char>(inputFile),
                          std::istreambuf_iterator<char>());

    while ((pos = this->raw_json.find('\n', pos)) != std::string::npos) {
      this->raw_json.erase(pos, 1);
    }

    block_allocator alloc(1 << 10);
    char json_src[this->raw_json.size() + 1];
    strcpy(json_src, this->raw_json.c_str());
    json_value *json = parseJSON(json_src, &alloc);

    if (json != NULL) {

      this->name.assign("dbn");

      json_value* j_struct = NULL;
      this->up_down_epochs = 10;
      this->recon_epochs = 10;
      this->batch_size = batch_size > 0 ? batch_size : 10;

      for (json_value *it = json->first_child; it; it = it->next_sibling) {
        if (::strcmp(it->name, "up_down_epochs") == 0) {
          this->up_down_epochs = it->int_value;
        } else if (::strcmp(it->name, "batch_size") == 0) {
          this->batch_size = it->int_value;
        } else if (::strcmp(it->name, "structure") == 0) {
          j_struct = it;
        } else if (::strcmp(it->name, "name") == 0) {
          this->name.assign(it->string_value);
        }
      }

      if (j_struct == NULL) {
        std::cerr << "No 'structure' key, cannot continue..." << std::endl;
      } else {

        if (json_path.find_last_of('/') == std::string::npos)
          this->rootdir = "./";
        else
          this->rootdir = json_path.substr(0, json_path.find_last_of('/'));

        this->root = this->init_tree(j_struct, io, this->batch_size);
        this->associative_memory = this->root->value();
      }

    } else {
      std::cerr << "Unable to read the DBN JSON config at " << json_path
                << ", continuing..." << std::endl;
    }
  }
}

void DeepBeliefNet::input_is_exhausted() {
  this->io_exhausted = true;
}

Tree<DeepBeliefNet::spNode>::spTree DeepBeliefNet::init_tree(
    json_value *json, DataIO<ailab::decimal_t>& io, size_t batch_size) {

  size_t width_of_leaf = 0;
  spNode dbnnode(new Node(this));
  Tree<spNode>::spTree tree(new Tree<spNode>(dbnnode));

  dbnnode->tree = tree.get();
  auto nchildren = tree->children.size();

  json_value* jdef = NULL;

  // Just a constant, in case the batch size is after the output...
  this->batch_size = batch_size;

  // This loop just parses the JSON and inserts children
  for (json_value *it = json->first_child; it != NULL; it = it->next_sibling) {
    if (::strcmp(it->name, "rbm") == 0) {
      jdef = it;
    } else if (::strcmp(it->name, "children") == 0) {
      size_t offset = 0;
      for (json_value* jv = it->first_child; jv != NULL; jv =
          jv->next_sibling) {
        auto subtree = this->init_tree(jv, io, batch_size);
        nchildren = tree->children.size();
        tree->add_child(subtree);
        assert(tree->children.size() == (nchildren + 1));
        subtree->value()->parent_column_offset = offset;
        offset += subtree->value()->rbm.hidden->data.cols();
      }

    } else if (::strcmp(it->name, "output") == 0 && (it->int_value != 0)) {
      // Node with output (Can be anywhere)
      this->output_nodes.push_back(dbnnode);
      dbnnode->output = io.getDrainer(it->string_value);

    } else if (::strcmp(it->name, "input") == 0 && (it->int_value != 0)) {
      // Input node, guaranteed to be a leaf
      dbnnode->input = io.getFiller(it->string_value);
      size_t data_width = io.width(it->string_value);
      assert(data_width == width_of_leaf || width_of_leaf == 0);
      width_of_leaf = data_width;

    } else if (::strcmp(it->name, "holdout") == 0 && (it->int_value != 0)) {
      // Mirroring input, this is only useful for leaves
      dbnnode->holdout = io.getFiller(it->string_value);
      size_t holdout_width = io.width(it->string_value);
      assert(holdout_width == width_of_leaf || width_of_leaf == 0);
      width_of_leaf = holdout_width;

    } else if (::strcmp(it->name, "batch_size") == 0) {
      it->int_value = this->batch_size;
    }
  }

  if (jdef == NULL) {
    std::cerr << "Unable to read definition for RBM" << std::endl;
    exit(-1);
  }

  /* Now we can initialize this DBN Node */

  nchildren = tree->children.size();

  if (tree->children.size() > 0) {
    std::vector<RBM::spUnitBlock> vis_blocks;
    RBM::spLayer new_vis(new RBM::Layer(this->context));

    for (Tree<spNode>::spTree child : tree->children) {
      DeepBeliefNet::spNode node = child->value();
      std::vector<RBM::spUnitBlock>& child_blocks = node->rbm.hidden->blocks;
      vis_blocks.insert(vis_blocks.end(), child_blocks.begin(),
                        child_blocks.end());
    }

    new_vis->init_blocks_vec(vis_blocks
                             , &(dbnnode->rbm.rng_states)
                             , this->batch_size);

    assert(new_vis->data.rows() == this->batch_size);
    assert(new_vis->data.cols() > 0);
    assert(new_vis->data.cols() == new_vis->recon.cols());

    dbnnode->rbm.init(jdef, this->rootdir, batch_size, new_vis);

    assert(new_vis->data.rows() == this->batch_size);
    assert(new_vis->data.cols() > 0);
    assert(new_vis->data.cols() == new_vis->recon.cols());


  } else {
    dbnnode->rbm.init(jdef, this->rootdir, batch_size, NULL);
    dbnnode->rbm.init_with_visible(width_of_leaf);
    dbnnode->rbm.verify_ready();
    this->leaf_nodes.push_back(dbnnode);
  }

  dbnnode->rbm.output_options = this->output_options;
  dbnnode->rbm.name = this->name + ":" + dbnnode->rbm.name;

  // load a saved version of this RBM
  dbnnode->rbm.load(this->rootdir);

  return tree;
}

void DeepBeliefNet::train() {
  this->verify_ready();

  if (this->output_options.log_per_n_batches > 0) {
    this->logSetup();
  }

  /* Phase 1: Greedy training */
  this->do_greedy_training();

  /* Phase 2: Up-Down training */
  this->do_backprop_training();
}

void DeepBeliefNet::do_backprop_training() {

  size_t logPerN = this->output_options.log_per_n_batches;
  if (logPerN == 0 && this->output_options.update_cli) {
    logPerN = 13;
  }

  for (size_t e = 0; e < this->up_down_epochs; e++) {
    auto tstart = Clock::now();
    size_t batch = 0;

    if (this->output_options.update_cli && (batch % logPerN) == 0) {
      std::clog << "Backprop on epoch " << (e + 1) << " of "
                << this->up_down_epochs << " ..." << std::endl;
    }

    this->io_exhausted = false;

    while (!this->io_exhausted) {

      this->run_batch();
      this->root->apply_root_first([&](Tree<spNode>* x) {
        RBM& rbm = x->value()->rbm;
        rbm.update();
        rbm.logHistograms(this->output_options.hist_bin_count, true);
        rbm.logError(rbm.error(), std::numeric_limits<decimal_t>::quiet_NaN());
      });

      if (this->output_options.update_cli && (batch % logPerN) == 0) {
        std::clog << "\r\tBatch: " << std::setw(10) << batch << std::flush;
      }
      batch++;
    }

    if (this->output_options.update_cli) {

      double microseconds = (std::chrono::duration_cast<
          std::chrono::microseconds>((Clock::now() - tstart))).count();

      std::cerr << "\nDone in " << (microseconds / 1000000) << "s" << std::endl;
    }
  }
}

void DeepBeliefNet::do_greedy_training() {

  bool all_rbms_are_tied = true;
  this->root->apply_root_first([&](Tree<spNode>* x)
  {
    all_rbms_are_tied &= x->value()->rbm.weights_are_symmetric();;
  });

  if (all_rbms_are_tied) {
    std::cerr << "Doing greedy training" << std::endl;

    this->root->apply_leaves_first(
        [this](Tree<spNode>* x)
        {
          size_t logFreq = this->output_options.log_per_n_batches;


          spNode n = x->value();
          RBM& rbm = n->rbm;

          rbm.logSetup(rbm.epochs);

          if( !rbm.fixed_weights )
          {

            for(size_t e=0; e < rbm.epochs; e++)
            {
              Instant tstart = Clock::now();
              StatCounter epochDataError;
              StatCounter epochHoldoutError;

              if(this->output_options.update_cli)
              {
                std::clog << "\t" << rbm.name << ": epoch " << (e+1) << " of " << rbm.epochs << std::endl;
              }

              this->io_exhausted = false;
              // This is one epoch
              size_t batch=0;

              while( !this->io_exhausted )
              {
                this->fill_visible_for_training(x);
                rbm.run(true);
                rbm.update();

                epochDataError.push( rbm.error() );

                if(logFreq > 0 && (batch % logFreq == 0) )
                {
                  rbm.trainingLog(batch, epochDataError, epochHoldoutError);
                } else if(this->output_options.update_cli && (batch % 15 == 0)) {
                  rbm.trainingCLILog(batch, epochDataError, epochHoldoutError);
                }

                batch++;
              }

              if( (logFreq > 0) && epochDataError.nObs() > 0)
              {
                rbm.logMeanError(epochDataError, epochHoldoutError);
              }

              if(this->output_options.update_cli)
              {
                double epoch_time = (Clock::now() - tstart).count();
                epoch_time /= 1000000000;

                std::clog << "\nEpoch done in " << std::setprecision(6) << epoch_time << "s" << std::endl;
              }
            }
          }
        });
  }
}

void DeepBeliefNet::fill_visible_for_training(Tree<spNode>* x) {
  if (x->children.size() > 0) {
    RBM &rbm = x->value()->rbm;

    assert(rbm.visible->isInitialized());
    assert(rbm.hidden->isInitialized());

    for (auto subTree : x->children) {
      spNode child = subTree->value();
      this->fill_visible_for_training(subTree.get());
      child->rbm.run(true);
      rbm.visible->data.read_rect(child->rbm.hidden->recon
          // Source first
          , 0, 0
          // Now Destination
          , 0, child->parent_column_offset);
    }

  } else {
    spNode node = x->value();
    assert(x->value()->input != NULL);
    try {
      node->input(node->rbm.visible->data);
    } catch (Exception::IOReachedEnd& e) {
      this->input_is_exhausted();
    }
  }
}

void DeepBeliefNet::logSetup() {
  std::stringstream ss;
  ss << ",\"dbn\":\"" << this->name << '"';
  ss << ",\"rbms\":[\"\"";

  this->root->apply_leaves_first([&ss](Tree<spNode>* x) {
    ss << ",\"" << x->value()->rbm.name << "\"";
  });

  ss << "]";

  ailab::Logger::log("jlog", ss);
}

void DeepBeliefNet::reconstruct() {
  this->io_exhausted = false;
  size_t logFreq = this->output_options.log_per_n_batches;

  if (logFreq == 0) {
    logFreq = 15;
  }

  this->root->apply_root_first([&](Tree<spNode>* x)
  {
    x->value()->rbm.params.gibbs = 0;
  });

  if (this->output_options.update_cli) {
    std::cout << "Reconstructing..." << std::endl;
  }

  size_t written_rows = 0;
  size_t batch = 0;

  while (!this->io_exhausted) {

    this->run_batch();
    for (spNode node : this->output_nodes) {
      written_rows += node->rbm.visible->recon.rows();
      node->output(node->rbm.visible->recon);
    }

    if (this->output_options.update_cli && (batch % logFreq == 0)) {
      std::cout << "\r\tBatch " << std::right << std::setw(10) << batch
                << std::flush;
    }
    batch++;
  }

  if (this->output_options.update_cli) {
    std::cout << "\n\tdone after writing " << written_rows << " rows."
              << std::endl;
  }

}

void DeepBeliefNet::write_energy(std::ostream& output) {
  size_t i = 0;
  ailab::RBM::Matrix energy(NULL);
  this->io_exhausted = false;

  this->root->apply_leaves_first([&](Tree<spNode>* x) {
    if( i++ > 0 ) output << ",";
    output << x->value()->rbm.name;
  });
  output << "\n";
  energy.init( { this->batch_size, i });

  while (!this->io_exhausted) {
    i = 0;
    this->run_batch();
    this->root->apply_leaves_first([&](Tree<spNode>* x) {
      RBM &rbm = x->value()->rbm;
      energy.read_col(rbm.get_visible_energy(true), 0, i++);
    });

    for (size_t row = 0; row < energy.rows(); row++) {
      output << energy.at(row, 0);
      for (size_t col = 1; col < energy.cols(); col++) {
        output << "," << energy.at(row, col);
      }
      output << "\n";
    }
  }
}

void DeepBeliefNet::run_batch() {
  this->associative_memory->run_toward_root();
  this->associative_memory->run_toward_leaves();
}

void DeepBeliefNet::verify_ready() {
  this->root->apply_leaves_first([](Tree<spNode>* x) {
    RBM &rbm = x->value()->rbm;

    if( !rbm.visible->isInitialized() )
    {
      if( !x->value()->isLeaf() )
      {
        size_t make_width = 0;

        for(auto subTree: x->children)
        {
          make_width += subTree->value()->rbm.hidden->data.cols();
        }
        rbm.init_with_visible(make_width);
        rbm.verify_ready();
      } else {
        // Leaves should be initialized by now...
      assert(false);
    }
  }
});
}

void DeepBeliefNet::save() {
  this->root->apply_root_first([&](Tree<spNode>* x) {
    x->value()->rbm.save();
  });
}

}  // End of the AILAB namespace
