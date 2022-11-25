#pragma once

#include <iostream>
#include <random>
#include <functional>
#include <chrono>
#include <algorithm>
#include <numbers>
#include <fstream>
#include "Eigen/Dense"
#include "Eigen/Sparse"
#include "unsupported/Eigen/CXX11/Tensor"
#include "Spectra/SymEigsSolver.h"

using namespace Eigen;
using namespace std;
using namespace Spectra;

typedef function<MatrixXd(int, int)> MatrixGen;
typedef function<double(VectorXd, VectorXd)> KernelFunc;
typedef function<double(MatrixXd, double, double, double)> SSolver;
typedef vector<pair<string, SSolver>> SMethods;
typedef function<double(MatrixXd, double, int)> LSolver;
typedef vector<pair<string, LSolver>> LMethods;
typedef function<ArrayXi(MatrixXd, ArrayXi, MatrixXi, ArrayXi, int)> FSolver;
typedef vector<pair<string, FSolver>> FMethods;
typedef pair<MatrixXd, ArrayXi> FDataset;

// Set the seed for random number generator
void set_seed(unsigned int seed);

// Generate random normal and rademacher matrices
MatrixGen gen_norm(double mu, double sigma);
MatrixGen gen_radem();

// Kernel functions
KernelFunc kernel_class();
KernelFunc kernel_gaussian(double gamma);

// Calculate matrix-based R\'enyi's entropy via eigenvalue decomposition
SSolver solve_eigen();

// Calculate low-rank R\'enyi's entropy via eigenvalue decomposition
LSolver solve_lowrank();
// Approximate low-rank R\'enyi's entropy via Lanczos iteration
LSolver solve_lanczos(int s);
// Approximate low-rank R\'enyi's entropy via implicit restarting Lanczos implemented by Spectra
LSolver solve_lanczos();
// Approximate low-rank R\'enyi's entropy via Gaussian random projection
LSolver solve_gaussian_proj(int s);
// Approximate low-rank R\'enyi's entropy via subsampled randomized Hadamard transform
LSolver solve_hadamard_proj(int s);
// Approximate low-rank R\'enyi's entropy via input-sparsity transform
LSolver solve_sparsity_proj(int s);
// Approximate low-rank R\'enyi's entropy via sparse graph sketching
LSolver solve_bipartite_proj(int s);

// Datasets for feature selection
FDataset load_breast();
FDataset load_semeion();
FDataset load_madelon();
FDataset load_krvskp();
FDataset load_spambase();
FDataset load_waveform();
FDataset load_optdigits();
FDataset load_statlog();

// Shannon's entropy-based feature selection methods
FSolver solve_mifs(double beta);
FSolver solve_fou();
FSolver solve_mrmr();
FSolver solve_jmi();
FSolver solve_cmim();
FSolver solve_disr();

// R\'enyi's entropy-based feature selection methods
FSolver solve_renyi_feature(double alpha, KernelFunc kernel, SSolver method); // MRMI
FSolver solve_lowrank_feature(double alpha, int k, KernelFunc kernel, LSolver method); // LRMI

// Feature selection based on greedy mutual information maximization
void run_features(MatrixXd x, ArrayXi type, int b, int t, bool use_timer, FMethods& methods);