#include "pch.h"

default_random_engine e((int) time(nullptr));

void set_seed(unsigned int seed) {
	e.seed(seed);
}

template<typename T>
MatrixGen generator(T& gen) {
	return [&gen](int rows, int cols) { return MatrixXd::NullaryExpr(rows, cols,
		[&gen]() { return gen(e); }); };
}

MatrixGen gen_norm(double mu, double sigma) {
	return generator(*new normal_distribution<>(mu, sigma));
}

MatrixGen gen_radem() {
	static uniform_int_distribution<> r(0, 1);
	return [](int rows, int cols) { return MatrixXd::NullaryExpr(rows, cols,
		[]() { return r(e) * 2 - 1; }); };
}

KernelFunc kernel_class() {
	return [](VectorXd x, VectorXd y) {
		return int(x(0) == y(0));
	};
}

KernelFunc kernel_gaussian(double sigma) {
	double gamma = -1 / (2 * sigma * sigma);
	return [gamma](VectorXd x, VectorXd y) {
		return exp((x - y).squaredNorm() * gamma);
	};
}

double calc_lowrank_entropy(MatrixXd mat, double alpha, int k) {
	int n = (int) mat.rows();
	SelfAdjointEigenSolver<MatrixXd> es(mat, EigenvaluesOnly);
	VectorXd evs = es.eigenvalues();
	double tr = evs.tail(k).array().cwiseMax(0).pow(alpha).sum();
	//cout << evs.reverse().transpose() << endl;
	if (k < n)
		tr += (n - k) * pow(max(0.0, evs.head(n - k).mean()), alpha);
	return log2(tr) / (1 - alpha);
}

MatrixXd kernel_matrix(MatrixXd x, MatrixXd y, KernelFunc kernel) {
	MatrixXd mat(x.rows(), y.rows());
	for (int i = 0; i < x.rows(); i++)
		for (int j = 0; j < y.rows(); j++)
			mat(i, j) = kernel(x.row(i), y.row(j));
	return mat;
}

MatrixXd normalized_kernel_matrix(MatrixXd x, MatrixXd y, KernelFunc kernel) {
	MatrixXd mat = kernel_matrix(x, y, kernel);
	return mat / mat.trace();
}

SSolver solve_eigen() {
	return [](MatrixXd K, double alpha, double u, double v) {
		SelfAdjointEigenSolver<MatrixXd> es(K, EigenvaluesOnly);
		return log2(es.eigenvalues().array().cwiseMax(0).pow(alpha).sum()) / (1 - alpha);
	};
}

double solve_projection(MatrixXd P, double alpha, int k) {
	BDCSVD<MatrixXd> svd(P);
	//cout << svd.singularValues().transpose() << endl;
	VectorXd svs = svd.singularValues();
	int n = (int) P.rows();
	double tr = svs.head(k).array().cwiseMax(0).pow(alpha).sum();
	//cout << svs.head(k).transpose() << endl;
	if (k < n)
		tr += (n - k) * pow(max(0.0, (1 - svs.head(k).sum()) / (n - k)), alpha);
	return log2(tr) / (1 - alpha);
}

LSolver solve_lowrank() {
	return [](MatrixXd K, double alpha, int k) {
		return calc_lowrank_entropy(K, alpha, k);
	};
}

LSolver solve_lanczos(int s) {
	return [s](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();

		VectorXd p = VectorXd::Zero(n); p(0) = 1;
		MatrixXd Q(s, n), T = MatrixXd::Zero(s, s);

		double beta = 0, beta0 = p.norm();
		Q.row(0) = p / beta0;
		for (int j = 0; j < s; j++) {
			VectorXd q = Q.row(j) * K;
			if (j > 0)
				q -= Q.row(j - 1) * beta;
			double gamma = Q.row(j).dot(q);
			T(j, j) = gamma;
			q -= Q.row(j) * gamma;
			for (int k = 0; k < j; k++)
				q -= Q.row(k).dot(q) * Q.row(k);
			beta = q.norm();
			if (j + 1 < s) {
				T(j, j + 1) = T(j + 1, j) = beta;
				Q.row(j + 1) = q / beta;
			}
		}

		SelfAdjointEigenSolver<MatrixXd> es(T, ComputeEigenvectors);
		VectorXd evs = es.eigenvalues();
		double tr = evs.tail(k).array().cwiseMax(0).pow(alpha).sum();
		//cout << evs.reverse().transpose() << endl;
		if (k < n)
			tr += (n - k) * pow(max(0.0, (1 - evs.tail(k).sum()) / (n - k)), alpha);
		return log2(tr) / (1 - alpha);
	};
}

LSolver solve_lanczos() {
	return [](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();
		DenseSymMatProd<double> op(K);
		SymEigsSolver<DenseSymMatProd<double>> eigs(op, k, k + 50);
		//cout << "eigen start" << endl;

		eigs.init();
		eigs.compute(SortRule::LargestAlge, 0, 1e-4);

		Eigen::VectorXd evalues = eigs.eigenvalues();
		double tr = evalues.array().cwiseMax(0).pow(alpha).sum();
		if (k < n)
			tr += (n - k) * pow(max(0.0, (1 - evalues.sum()) / (n - k)), alpha);
		return log2(tr) / (1 - alpha);
	};
}

LSolver solve_gaussian_proj(int s) {
	static MatrixGen gen_g = gen_norm(0, 1);
	return [s](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();
		MatrixXd P = gen_g(n, s);
		for (int i = 0; i < s; i++) {
			for (int j = 0; j < i; j++)
				P.col(i) -= P.col(i).dot(P.col(j)) * P.col(j);
			P.col(i).normalize();
		}
		return solve_projection(K * P * sqrt((double) n / s), alpha, k);
	};
}

LSolver solve_hadamard_proj(int s) {
	static MatrixGen gen_r = gen_radem();
	return [s](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();
		MatrixXd P(n, s);
		VectorXd R = gen_r(n, 1);
		uniform_int_distribution<> smp(0, n - 1);

		for (int i = 0; i < s; i++) {
			uint32_t k = smp(e);
			for (int j = 0; j < n; j++)
				P(j, i) = popcount(j & k) % 2 == 0 ? 1 : -1;
		}
		for (int i = 0; i < n; i++)
			P.row(i) *= R(i);
		return solve_projection(K * P / sqrt(s), alpha, k);
	};
}

LSolver solve_sparsity_proj(int s) {
	static MatrixGen gen_r = gen_radem();
	return [s](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();
		MatrixXd P(n, s);
		VectorXd R = gen_r(s, 1);
		uniform_int_distribution<> smp(0, n - 1);
		for (int i = 0; i < s; i++)
			P.col(i) = R(i) * K.col(smp(e));
		return solve_projection(P * sqrt((double) n / s), alpha, k);
	};
}

LSolver solve_bipartite_proj(int s) {
	static MatrixGen gen_r = gen_radem();
	return [s](MatrixXd K, double alpha, int k) {
		int n = (int) K.rows();
		auto gen_c = uniform_int_distribution<>(0, s - 1);
		SparseMatrix<double> P(n, s);
		MatrixXd R = gen_r(n, 2) / sqrt(2);
		for (int i = 0; i < n; i++) {
			int c0 = gen_c(e), c1 = gen_c(e);
			while (c1 == c0)
				c1 = gen_c(e);
			P.insert(i, c0) = R(i, 0);
			P.insert(i, c1) = R(i, 1);
		}
		return solve_projection(K * P, alpha, k);
	};
}

VectorXd calc_dist(VectorXi x, int xc) {
	VectorXd res(xc);
	res.setZero();
	for (int i = 0; i < x.size(); i++)
		res(x(i))++;
	return res / res.sum();
}

MatrixXd calc_dist(VectorXi x, int xc, VectorXi y, int yc) {
	MatrixXd res(xc, yc);
	res.setZero();
	for (int i = 0; i < x.size(); i++)
		res(x(i), y(i))++;
	return res / res.sum();
}

Tensor<double, 3> calc_dist(VectorXi x, int xc, VectorXi y, int yc, VectorXi z, int zc) {
	Tensor<double, 3> res(xc, yc, zc);
	res.setZero();
	for (int i = 0; i < x.size(); i++)
		res(x(i), y(i), z(i))++;
	Tensor<double, 0> sum = res.sum();
	return res / sum();
}

double calc_e(VectorXi x, int xc) {
	VectorXd px = calc_dist(x, xc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		if (px(i) > 0)
			res -= px(i) * log(px(i));
	return res;
}

double calc_je(VectorXi x, int xc, VectorXi y, int yc) {
	MatrixXd pxy = calc_dist(x, xc, y, yc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			if (pxy(i, j) > 0)
				res -= pxy(i, j) * log(pxy(i, j));
	return res;
}

double calc_je(VectorXi x, int xc, VectorXi y, int yc, VectorXi z, int zc) {
	Tensor<double, 3> pxyz = calc_dist(x, xc, y, yc, z, zc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			for (int k = 0; k < zc; k++)
				if (pxyz(i, j, k) > 0)
					res -= pxyz(i, j, k) * log(pxyz(i, j, k));
	return res;
}

double calc_wje(VectorXi x, int xc, VectorXi y, int yc) {
	VectorXd px = calc_dist(x, xc);
	MatrixXd pxy = calc_dist(x, xc, y, yc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			if (pxy(i, j) > 0)
				res -= px(i) * pxy(i, j) * log(pxy(i, j));
	return res;
}

double calc_mi(VectorXi x, int xc, VectorXi y, int yc) {
	VectorXd px = calc_dist(x, xc);
	VectorXd py = calc_dist(y, yc);
	MatrixXd pxy = calc_dist(x, xc, y, yc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			if (pxy(i, j) > 0)
				res += pxy(i, j) * log(pxy(i, j) / px(i) / py(j));
	return res;
}

double calc_mi(VectorXi x, int xc, VectorXi y, int yc, VectorXi z, int zc) {
	VectorXd pz = calc_dist(z, zc);
	MatrixXd pxy = calc_dist(x, xc, y, yc);
	Tensor<double, 3> pxyz = calc_dist(x, xc, y, yc, z, zc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			for (int k = 0; k < zc; k++)
				if (pxyz(i, j, k) > 0)
					res += pxyz(i, j, k) * log(pxyz(i, j, k) / pxy(i, j) / pz(k));
	return res;
}

double calc_cmi(VectorXi x, int xc, VectorXi y, int yc, VectorXi z, int zc) {
	VectorXd pz = calc_dist(z, zc);
	MatrixXd pxz = calc_dist(x, xc, z, zc);
	MatrixXd pyz = calc_dist(y, yc, z, zc);
	Tensor<double, 3> pxyz = calc_dist(x, xc, y, yc, z, zc);

	double res = 0;
	for (int i = 0; i < xc; i++)
		for (int j = 0; j < yc; j++)
			for (int k = 0; k < zc; k++)
				if (pxyz(i, j, k) > 0)
					res += pxyz(i, j, k) * log(pz(k) * pxyz(i, j, k) / pxz(i, k) / pyz(j, k));
	return res;
}

FDataset load_breast() {
	ifstream fin("dataset/breast/wdbc.txt");

	int n = 569, d = 31;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.fill(1);
	type(30) = 0;
	return { data, type };
}

FDataset load_semeion() {
	ifstream fin("dataset/semeion/semeion.txt");

	int n = 1593, d = 257;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.setZero();
	return { data, type };
}

FDataset load_madelon() {
	ifstream fin("dataset/madelon.txt");

	int n = 2600, d = 501;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.setZero();
	return { data, type };
}

FDataset load_krvskp() {
	ifstream fin("dataset/krvskp/krvskp.txt");

	int n = 3196, d = 37;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.setZero();
	type(14) = 2;
	return { data, type };
}

FDataset load_spambase() {
	ifstream fin("dataset/spambase/train.txt");

	int n = 4601, d = 58;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.head(56).fill(1);
	type.tail(2).setZero();
	return { data, type };
}

FDataset load_waveform() {
	ifstream fin("dataset/waveform/train.txt");

	int n = 5000, d = 41;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.head(40).fill(1);
	type(40) = 0;
	return { data, type };
}

FDataset load_optdigits() {
	ifstream fin("dataset/optdigits/optdigits.txt");

	int n = 5620, d = 65;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.setZero();
	return { data, type };
}

FDataset load_statlog() {
	ifstream fin("dataset/statlog/statlog.txt");

	int n = 6435, d = 37;
	MatrixXd data(n, d);
	for (int i = 0; i < n; i++)
		for (int j = 0; j < d; j++)
			fin >> data(i, j);

	ArrayXi type(d);
	type.setZero();
	return { data, type };
}

FSolver solve_renyi_feature(double alpha, KernelFunc kernel, SSolver method) {
	return [alpha, kernel, method](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1, n = (int) x.rows();
		MatrixXd y(n, d + 1);
		chrono::high_resolution_clock hrc; auto ti = hrc.now();
		for (int i = 0; i < d + 1; i++) {
			double u = x.col(i).mean();
			double s = sqrt((x.col(i).array() - u).square().sum() / n);
			if (abs(s) < 1e-10)
				y.col(i).setZero();
			else y.col(i) = (x.col(i).array() - u) / s;
		}

#if 1
		vector<MatrixXd, aligned_allocator<MatrixXd>> K;
		K.resize(d + 1);
		for (int i = 0; i < d; i++) {
			K[i] = kernel_matrix(y.col(i), y.col(i), type(i) == 2 ? kernel_class() : kernel);
			cout << i << endl;
		}
		K[d] = kernel_matrix(y.col(d), y.col(d), kernel_class());

		auto calc = [&K, n](ArrayXi &cols) {
			MatrixXd mat = K[cols[0]];
			for (int i = 1; i < cols.size(); i++)
				mat = mat.cwiseProduct(K[cols[i]]);
			mat /= mat.trace();
			return mat;
		};
#define pre
#endif
		cout << "Phase 1" << ' ' << chrono::duration_cast<chrono::microseconds>(hrc.now() - ti).count() / 1000.0 << endl;
		ti = hrc.now();

		bool non_int = abs(alpha - round(alpha)) > 1e-4;
		ArrayXi res(t);
		for (int i = 0; i < t; i++) {
			double mi = -DBL_MAX;
			for (int j = 0; j < d; j++) {
				bool used = false;
				for (int k = 0; k < i; k++) {
					if (res(k) == j) {
						used = true;
						break;
					}
				}

				if (used) continue;
				double nmi = 0;
				{
					ArrayXi sel = res.head(i);
					sel.conservativeResize(sel.size() + 1);
					sel(sel.size() - 1) = j;
					{
#ifdef pre
						MatrixXd K = calc(sel);
#else
						MatrixXd K = normalized_kernel_matrix(y(all, sel), y(all, sel), kernel);
#endif
						nmi += method(K, alpha, 0, 1);
					}
					sel.conservativeResize(sel.size() + 1);
					sel(sel.size() - 1) = d;
					{
#ifdef pre
						MatrixXd K = calc(sel);
#else
						MatrixXd K = normalized_kernel_matrix(y(all, sel), y(all, sel), kernel);
#endif
						nmi -= method(K, alpha, 0, 1);
					}

					if (nmi > mi) {
						mi = nmi;
						res(i) = j;
					}

					cout << res.head(i + 1).transpose() << ' ' << j << ' ' << mi << ' ' << nmi << endl;
				}
			}
		}

		cout << "Phase 2" << ' ' << chrono::duration_cast<chrono::microseconds>(hrc.now() - ti).count() / 1000.0 << endl;
		return res;
	};
#ifdef pre
#undef pre
#endif
}

FSolver solve_lowrank_feature(double alpha, int k, KernelFunc kernel, LSolver method) {
	return [alpha, k, kernel, method](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1, n = (int) x.rows();
		MatrixXd y(n, d + 1);
		chrono::high_resolution_clock hrc; auto ti = hrc.now();
		ArrayXd var(d + 1);
		for (int i = 0; i < d + 1; i++) {
			double u = x.col(i).mean();
			double s = sqrt((x.col(i).array() - u).square().sum() / n);
			var(i) = s;
			if (abs(s) < 1e-10)
				y.col(i).setZero();
			else y.col(i) = (x.col(i).array() - u) / s;
		}

#if 1
		vector<MatrixXd, aligned_allocator<MatrixXd>> K;
		K.resize(d + 1);
		for (int i = 0; i < d; i++) {
			K[i] = kernel_matrix(y.col(i), y.col(i), type(i) == 2 ? kernel_class() : kernel);
			cout << i << endl;
		}
		K[d] = kernel_matrix(y.col(d), y.col(d), kernel_class());

		auto calc = [&K, n](ArrayXi &cols) {
			MatrixXd mat = K[cols[0]];
			for (int i = 1; i < cols.size(); i++)
				mat = mat.cwiseProduct(K[cols[i]]);
			mat /= mat.trace();
			return mat;
		};
#define pre
#endif
		cout << "Phase 1" << ' ' << chrono::duration_cast<chrono::microseconds>(hrc.now() - ti).count() / 1000.0 << endl;
		ti = hrc.now();

		bool non_int = abs(alpha - round(alpha)) > 1e-4;
		ArrayXi res(t);
		for (int i = 0; i < t; i++) {
			double mi = -DBL_MAX;
			for (int j = 0; j < d; j++) {
				bool used = false;
				for (int k = 0; k < i; k++) {
					if (res(k) == j) {
						used = true;
						break;
					}
				}

				if (used) continue;
				double nmi = 0;
				{
					ArrayXi sel = res.head(i);
					sel.conservativeResize(sel.size() + 1);
					sel(sel.size() - 1) = j;
					{
						bool zero_var = true;
						for (int f : sel)
							if (var(f) > 1e-10)
								zero_var = false;
						if (!zero_var) {
#ifdef pre
							MatrixXd K = calc(sel);
#else
							MatrixXd K = normalized_kernel_matrix(y(all, sel), y(all, sel), kernel);
#endif
							nmi += method(K, alpha, k);
						}
					}
					sel.conservativeResize(sel.size() + 1);
					sel(sel.size() - 1) = d;
					{
#ifdef pre
						MatrixXd K = calc(sel);
#else
						MatrixXd K = normalized_kernel_matrix(y(all, sel), y(all, sel), kernel);
#endif
						nmi -= method(K, alpha, k);
					}

					if (nmi > mi) {
						mi = nmi;
						res(i) = j;
					}

					cout << res.head(i + 1).transpose() << ' ' << j << ' ' << mi << ' ' << nmi << endl;
				}
			}
		}

		cout << "Phase 2" << ' ' << chrono::duration_cast<chrono::microseconds>(hrc.now() - ti).count() / 1000.0 << endl;
		return res;
	};
#ifdef pre
#undef pre
#endif
}

ArrayXi rank_feature(int d, int t, function<double(int)> calc_mi) {
	vector<pair<double, int>> rank;
	for (int i = 0; i < d; i++) {
		rank.push_back({ calc_mi(i), i });
		cout << i << '\t' << rank.rbegin()->first << endl;
	}
	sort(rank.begin(), rank.end(), std::greater<>());
	ArrayXi res(t);
	for (int i = 0; i < t; i++)
		res(i) = rank[i].second;
	return res;
}

ArrayXi select_feature(int d, int t, function<double(ArrayXi, int)> calc_mi) {
	ArrayXi res(t);
	for (int i = 0; i < t; i++) {
		double mi = -DBL_MAX;
		for (int j = 0; j < d; j++) {
			bool used = false;
			for (int k = 0; k < i; k++) {
				if (res(k) == j) {
					used = true;
					break;
				}
			}

			if (used) continue;
			double nmi = calc_mi(res.head(i), j);
			if (nmi > mi) {
				mi = nmi;
				res(i) = j;
			}
		}
	}

	return res;
}

FSolver solve_mifs(double beta) {
	return [beta](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d, beta](ArrayXi sel, int cur) {
			double mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d));
			for (int i = 0; i < sel.size(); i++)
				mi -= beta * calc_mi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)));
			return mi;
			});
	};
}

FSolver solve_fou() {
	return [](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d](ArrayXi sel, int cur) {
			double mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d));
			for (int i = 0; i < sel.size(); i++)
				mi -= (calc_mi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)))
					- calc_cmi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)), xd.col(d), xc(d)));
			return mi;
			});
	};
}

FSolver solve_mrmr() {
	return [](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d](ArrayXi sel, int cur) {
			double mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d));
			for (int i = 0; i < sel.size(); i++)
				mi -= calc_mi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i))) / sel.size();
			return mi;
			});
	};
}

FSolver solve_jmi() {
	return [](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d](ArrayXi sel, int cur) {
			double mi = 0;
			if (sel.size() == 0) {
				mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d));
			} else {
				for (int i = 0; i < sel.size(); i++)
					mi += calc_mi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)), xd.col(d), xc(d));
			}
			return mi;
			});
	};
}

FSolver solve_cmim() {
	return [](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d](ArrayXi sel, int cur) {
			double mi = DBL_MAX;
			if (sel.size() == 0) {
				mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d));
			} else {
				for (int i = 0; i < sel.size(); i++)
					mi = min(mi, calc_cmi(xd.col(cur), xc(cur), xd.col(d), xc(d), xd.col(sel(i)), xc(sel(i))));
			}
			return mi;
			});
	};
}

FSolver solve_disr() {
	return [](MatrixXd x, ArrayXi type, MatrixXi xd, ArrayXi xc, int t) {
		int d = (int) x.cols() - 1;
		return select_feature(d, t, [&xd, &xc, d](ArrayXi sel, int cur) {
			double mi = 0;
			if (sel.size() == 0) {
				mi = calc_mi(xd.col(cur), xc(cur), xd.col(d), xc(d)) / calc_je(xd.col(cur), xc(cur), xd.col(d), xc(d));
			} else {
				for (int i = 0; i < sel.size(); i++)
					mi += calc_mi(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)), xd.col(d), xc(d))
						/ calc_je(xd.col(cur), xc(cur), xd.col(sel(i)), xc(sel(i)), xd.col(d), xc(d));
			}
			return mi;
			});
	};
}

void make_discrete(MatrixXd &x, ArrayXi &type, int b, MatrixXi &xd, ArrayXi &xm) {
	int n = (int) x.rows(), d = (int) x.cols();

	for (int i = 0; i < d; i++) {
		if (type(i) == 1) {
			VectorXd col = x.col(i);
			sort(col.begin(), col.end());

			ArrayXd bins(b);
			for (int j = 1; j <= b; j++)
				bins(j - 1) = col(n * j / b - 1);
			for (int j = 0; j < n; j++)
				xd(j, i) = (int) distance(bins.begin(), lower_bound(bins.begin(), bins.end(), x(j, i)));
			xm(i) = b;
		} else {
			unordered_map<double, int> bins;
			for (int j = 0; j < n; j++) {
				double v = x(j, i);
				auto iter = bins.find(v);
				if (iter != bins.end())
					xd(j, i) = iter->second;
				else {
					xd(j, i) = (int) bins.size();
					bins.insert({ v, (int) bins.size() });
				}
			}

			xm(i) = (int) bins.size();
		}
	}
}

void run_features(MatrixXd x, ArrayXi type, int b, int t, bool use_timer, FMethods& methods) {
	int n = (int) x.rows(), d = (int) x.cols();

	MatrixXi xd(n, d);
	ArrayXi xm(d);
	make_discrete(x, type, b, xd, xm);

	for (auto method : methods) {
		chrono::high_resolution_clock hrc; auto ti = hrc.now();
		ArrayXi res = method.second(x, type, xd, xm, t);
		double timer = chrono::duration_cast<chrono::microseconds>(hrc.now() - ti).count() / 1000.0;
		cout << method.first << " [";
		for (int i : res)
			cout << i << ", ";
		cout << "] ";
		if (use_timer)
			cout << timer;
		cout << endl;
	}
}