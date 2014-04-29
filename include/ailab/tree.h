#ifndef TREE_H
#define TREE_H

#include <assert.h>
#include <algorithm>
#include <functional>
#include <memory>
#include <vector>

namespace ailab {

template<typename T>
class Tree {

 public:
  typedef std::shared_ptr<Tree<T> > spTree;

 protected:

  size_t _set_depths() {
    for (spTree x : this->children) {
      x->depth = this->depth + 1;
      this->height = std::max(this->height, x->_set_depths());
    }
    return this->height + 1;
  }

  T _value;
 public:

  Tree()
      : depth(0),
        height(0),
        parent(NULL) {
  }

  Tree(T v)
      : _value(v),
        depth(0),
        height(0),
        parent(NULL) {
  }

  Tree(T v, std::weak_ptr<Tree<T> > parent)
      : _value(v),
        depth(0),
        height(0),
        parent(parent) {
  }

  T value() {
    return this->_value;
  }
  void value(T v) {
    this->_value = v;
  }

  spTree add_child(T v) {
    this->children.push_back(spTree(new Tree(v, this)));
    this->children.back()->depth = this->depth + 1;
    return this->children.back();
  }

  spTree add_child(spTree n) {
    assert(n->parent == NULL);
    n->parent = this;
    n->depth = this->depth + 1;
    this->children.push_back(n);
    return n;
  }

  Tree<T>& apply_root_first(std::function<void(Tree<T>*)> function) {
    function(this);
    for (spTree t : this->children) {
      t->apply_root_first(function);
    }
    return *this;
  }

  Tree<T>& apply_leaves_first(std::function<void(Tree<T>*)> function) {
    for (spTree t : this->children) {
      t->apply_leaves_first(function);
    }

    function(this);

    return *this;
  }

  std::vector<T> leaf_values() {
    std::vector<T> values;
    if (this->children.size() > 0) {
      for (spTree n : this->children) {
        n->leaf_values(values);
      }
    } else {
      values.push_back(this->value());
    }

    return values;
  }

  void leaf_values(std::vector<T>& values) {
    if (this->children.size() > 0) {
      for (spTree n : this->children)
        n->leaf_values(values);
    } else {
      values.push_back(this->value());
    }
  }

  void set_depths() {
    this->depth = 0;
    this->height = 0;
    this->_set_depths();
  }

  size_t leaf_count() {
    size_t n_leaves = 0;

    this->apply_root_first([&n_leaves](Tree<T> * node) {
      if( node.children.size() == 0 ) {
        n_leaves++;
      }
    });

    return n_leaves;
  }

  Tree<T> * parent;
  std::vector<spTree> children;
  size_t depth;
  size_t height;
};

}

#endif // TREE_H
