#include "pch.h"

void run_features() {
	// Load a dataset
	FDataset dataset = load_madelon();

	FMethods methods = {
		{ "mifs", solve_mifs(0.7) },
		{ "fou", solve_fou() },
		{ "mrmr", solve_mrmr() },
		{ "jmi", solve_jmi() },
		{ "cmim", solve_cmim() },
		{ "disr", solve_disr() },
		{ "mrmi", solve_renyi_feature(2, kernel_gaussian(1), solve_eigen()) }, // MRMI with alpha = 2
		{ "lrmi", solve_lowrank_feature(2, 100, kernel_gaussian(1), solve_lanczos()) }, // LRMI with alpha = 2, k = 100
	};

	// Select the first 10 most informative features
	// For Shannon's entropy-based methods, continuous features are discretized in to 5 bins.
	run_features(dataset.first, dataset.second, 5, 10, true, methods);
}

int main(int argc, char** argv) {
	ios::sync_with_stdio(false);
	set_seed(0);

	run_features();
	return 0;
}
