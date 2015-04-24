#Tensor machines for learning target-specific polynomial features

Jiyan Yang (jiyan@stanford.edu)  

Alex Gittens (gittens@icsi.berkeley.edu)

##About
Tensor machines find a parsimonious set of polynomial features in a target-specific manner.
See the paper [Tensor machines for learning target-specific polynomial features](http://arxiv.org/pdf/1504.01697v1.pdf) for more details.

This repository contains a collection of codes used to train tensor machines on a given dataset and evaluate their generalization performance.

##Codes
The usage of each code is documented in the corresponding ```.m``` file. In particular,
- ```tensor_machines.m``` is the main file that implements tensor machines;
- ```tm_fg.m``` and ```tm_fg0.m``` evaluate the function objective and compute the gradient of the underlying optimization problem;
- ```get_tm_pred.m``` calculates prediction on a new dataset using a learned model after training;
- ```cv_tensor_machines.m``` uses grid search to tune hyper-parameters for tensor machines;
- ```tm_solver.m``` serves as an interface that allows one to train tensor machines and evaluate the generalization performance of tensor machines on a test set.

##Solvers
The underlying optimization solver is central in training tensor machines.
We considered two solvers: minFunc and SFO.
- [SFO](https://github.com/Sohl-Dickstein/Sum-of-Functions-Optimizer) is included in this package.
- minFunc can be downloaded via this [link](http://www.cs.ubc.ca/~schmidtm/Software/minFunc.html).

For the parameters used in these two solvers, see ```tensor_machines.m```.

##Datasets
In ```datasets/```, two publicly available datasets from UCI repository are included.
- ```adult.mat```: the [Adult dataset](http://archive.ics.uci.edu/ml/datasets/Adult);
- ```forest_small.mat```: a subset of the [Forest dataset](https://archive.ics.uci.edu/ml/datasets/Covertype); this has the test set of the original Forest dataset and a subset of 50000 training examples chosen randomly from the original training set.

##Examples
- ```main.m``` demonstrates how to use our code to fit and evaluate TMs on a real dataset.
- ```main2.m``` is similar to ```main.m``` but here the target is artificial, to allow us to test TMs performance on known polynomial targets

##Reference
Jiyan Yang and Alex Gittens, [Tensor machines for learning target-specific polynomial features](http://arxiv.org/pdf/1504.01697v1.pdf).

##License
The MIT License (MIT)

Copyright (c) 2015 Jiyan Yang and Alex Gittens

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in
all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
THE SOFTWARE.
