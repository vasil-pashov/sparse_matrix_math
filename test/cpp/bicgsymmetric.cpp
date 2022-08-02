#include "doctest/doctest.h"
#include "sparse_matrix_math.h"
#include "test_common.h"
#include <string>
#include <vector>

TEST_CASE_TEMPLATE("BiConjugate Gradient Symmetric method", T, float, double) {
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
	REQUIRE_EQ(SMM::BiCGSymmetric<T>(m, rhs, x, -1, l2Eps<T>()), SMM::SolverStatus::SUCCESS);

	for (const T ri : x) {
		CHECK_EQ(T(1), doctest::Approx(ri).epsilon(infEps<T>()));
	}
}