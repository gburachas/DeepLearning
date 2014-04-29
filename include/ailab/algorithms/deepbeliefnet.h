#ifndef DEEPBELIEFNET_H
#define DEEPBELIEFNET_H

#include <iostream>
#include <fstream>
#include <functional>
#include <stdint.h>
#include <string>
#include <memory>
#include <list>
#include <vector>
#include <set>
#include <map>
#include <algorithm>

#include <ailab/common.h>
#include <ailab/tree.h>
#include <ailab/opencl.h>
#include <ailab/algorithms/rbm.h>
#include <ailab/dataio.h>

namespace ailab {

/**
 * @brief The DeepBeliefNet class
 *
 * Citation: Hinton, Geoffrey E., Simon Osindero, and Yee-Whye Teh. "A fast learning algorithm for deep belief nets."
 *           Neural computation 18.7 (2006): 1527-1554.
 **/

class DeepBeliefNet : public CLBacked {
 public:

  class Node;
  typedef std::shared_ptr<Node> spNode;
  typedef std::function<void(Tree<spNode>* x)> TreeFunc;
  typedef std::function<void(OCLMatrix<decimal_t>&)> Spout;
  typedef std::function<void(OCLMatrix<decimal_t>&)> Drain;

  class Node {
   public:
    Node(DeepBeliefNet* net);

    void run_toward_root();
    void run_toward_leaves();bool isLeaf();bool isRoot();

    Spout input;
    Spout holdout;
    Drain output;

    size_t parent_column_offset;
    RBM rbm;
    Tree<spNode> * tree;
    DeepBeliefNet * dbn;
  };

  DeepBeliefNet(OpenCL::spContext context);
  void init(std::string json_path, DataIO<ailab::decimal_t>& io, size_t batch_size);

  void input_is_exhausted();

  void train();
  void reconstruct();
  void write_energy(std::ostream& output);
  void run_batch();

  void verify_ready();

  void save();

  size_t up_down_epochs;
  size_t recon_epochs;
  size_t batch_size;

  spNode associative_memory;
  Tree<spNode>::spTree root;

  std::string name;
  std::string rootdir;

  RBM::OutputOptions output_options;

 protected:

  Tree<spNode>::spTree init_tree(json_value* json,
                                 DataIO<ailab::decimal_t> &io,
                                 size_t batch_size);

  void do_greedy_training();
  void do_backprop_training();
  void fill_visible_for_training(Tree<spNode>* x);

  void logSetup();

  std::vector<spNode> leaf_nodes;
  std::vector<spNode> output_nodes;

  std::string raw_json;
  bool io_exhausted;

};

typedef std::shared_ptr<DeepBeliefNet> spDeepBeliefNet;

}  // End of the AILAB namespace
#endif // DEEPBELIEFNET_HPP
