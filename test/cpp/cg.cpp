#include "doctest/doctest.h"
#include "sparse_matrix_math.h"
#include "test_common.h"
#include <string>
#include <vector>

TEST_CASE_TEMPLATE("Conjugate Gradient method", T, float, double) {
	std::string matrixName;
	const std::vector<std::string> matrixNames = {
		"mesh1e1_structural_48_48_177.mtx",
		"mesh1em1_structural_48_48_177.mtx",
		"mesh1em6_structural_48_48_177.mtx"
	};
	DOCTEST_VALUE_PARAMETERIZED_DATA(matrixName, matrixNames);

	const std::string matrixPath = getMatrixPath(matrixName);
	SMM::CSRMatrix<T> m;
	REQUIRE_EQ(SMM::loadMatrix(matrixPath.c_str(), m), SMM::MatrixLoadStatus::SUCCESS);
	SMM::Vector<T> rhs = sumColumsPerRow(m);
	SMM::Vector<T> x(m.getDenseRowCount(), 0);
	REQUIRE_EQ(SMM::ConjugateGradient<T>(m, rhs, x, x, -1, l2Eps<T>()), SMM::SolverStatus::SUCCESS);

	for (const T ri : x) {
		CHECK_EQ(T(1), doctest::Approx(ri).epsilon(infEps<T>()));
	}
}

TEST_CASE_TEMPLATE("Compute and apply IC0", T, float, double) {
	const int size = 5;
	// {10, 0, 0 , 4 , 0},
	// {0 , 9, 0 , 0 , 5},
	// {0 , 0, 12, 0 , 0},
	// {4 , 0, 0 , 15, 7},
	// {0 , 5, 0 , 7 , 8}
	SMM::TripletMatrix<T> triplet(size, size);
	triplet.addEntry(0, 3, 4);
	triplet.addEntry(0, 0, 10);
	triplet.addEntry(1, 1, 9);
	triplet.addEntry(1, 4, 5);
	triplet.addEntry(2, 2, 12);
	triplet.addEntry(3, 0, 4);
	triplet.addEntry(3, 3, 15);
	triplet.addEntry(3, 4, 7);
	triplet.addEntry(4, 1, 5);
	triplet.addEntry(4, 3, 7);
	triplet.addEntry(4, 4, 8);

	SMM::CSRMatrix<T> m;
	m.init(triplet);
	SMM::CSRMatrix<T>::IC0Preconditioner ic0(m);
	REQUIRE_EQ(ic0.init(), 0);
	T rhs[size];
	std::fill_n(rhs, size, 1);
	T res[size];
	T resRef[size] = {0.0995763, 0.0646186, 0.0833333, 0.0010593, 0.0836864};
	ic0.apply(rhs, res);
	for (int i = 0; i < size; ++i) {
		CHECK_EQ(doctest::Approx(res[i]).epsilon(1e-4), resRef[i]);
	}
}

TEST_CASE_TEMPLATE("Preconditioned Conjugate Gradient method. IC0 Preconditioner", T, float, double) {
	std::string matrixName;
	const std::vector<std::string> matrixNames = {
		"mesh1e1_structural_48_48_177.mtx",
		"mesh1em1_structural_48_48_177.mtx",
		"mesh1em6_structural_48_48_177.mtx"
	};
	DOCTEST_VALUE_PARAMETERIZED_DATA(matrixName, matrixNames);

	const std::string matrixPath = getMatrixPath(matrixName);
	SMM::CSRMatrix<T> m;
	REQUIRE_EQ(SMM::loadMatrix(matrixPath.c_str(), m), SMM::MatrixLoadStatus::SUCCESS);
	SMM::Vector<T> rhs = sumColumsPerRow(m);
	SMM::Vector<T> x(m.getDenseRowCount(), 0);

	typename SMM::CSRMatrix<T>::IC0Preconditioner M(m);
	REQUIRE_EQ(M.init(), 0);
	REQUIRE_EQ(SMM::ConjugateGradient<T>(m, rhs, x, x, -1, l2Eps<T>(), M), SMM::SolverStatus::SUCCESS);

	for (const T ri : x) {
		CHECK_EQ(T(1), doctest::Approx(ri).epsilon(infEps<T>()));
	}
}