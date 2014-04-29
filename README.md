DeepLearning
============
An OpenCL implementation of Deep Belief Networks and Restricted Boltzmann Machines

Dependencies
---------
This project depends on the vjson library, which is out-dated and needs to be replaced
The gflags library is also required.

As a first step you need to edit rbm/Makefile and dbn/Makefile to point to your OpenCL
implementation. After that, just run make in either of these directories to build that
project. Updates to the RBM code have caused the DBNs to not compile, so beware.
