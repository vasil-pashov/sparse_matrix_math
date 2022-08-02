#if 0

#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

template<typename T>
class BiCGStab : public SolverTest<T> {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps);
	}
};

template<typename T>
class BiCGStab_PositiveDefinite : public BiCGStab<T> {

};

template<typename T>
class BiCGStab_Indefinite : public BiCGStab<T> {

};


using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(BiCGStab, MyTypes);
TYPED_TEST_SUITE(BiCGStab_PositiveDefinite, MyTypes);
TYPED_TEST_SUITE(BiCGStab_Indefinite, MyTypes);

TYPED_TEST(BiCGStab, SmallDenseMatrix) {
	GTEST_SKIP_("The numerical instability of the vector is clearly seen here. Using fma helps a little, but not ehough.");
	SMM::TripletMatrix<TypeParam> triplet(4, 4, 16);
	TypeParam dense[16] = { 30.49, 13.95, 9.6, 15.75, 13.95, 18.83, 4.93, 12.91, 9.6, 4.93, 11.89, 0.68, 15.75, 12.91, 0.68, 13.41 };
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			triplet.addEntry(row, col, dense[row * 4 + col]);
		}
	}

	SMM::CSRMatrix<TypeParam> csr(triplet);
	SMM::Vector<TypeParam> b({1,2,3,4});
	SMM::Vector<TypeParam> x(4, 0);

	EXPECT_EQ(this->solve(csr, b, x, -1, L2Epsilon<TypeParam>()), SMM::SolverStatus::SUCCESS);
	TypeParam resRef[4] = { -5.57856, -5.62417, 6.40556, 11.9399 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(x[i], resRef[i], MaxInfEpsilon<TypeParam>());
	}
}

TYPED_TEST(BiCGStab_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStab_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStab_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStab_Indefinite, sherman1_1000_1000_2375) {
	GTEST_SKIP_("The numerical instability of the method is clearly seen here. Using fma helps a little, but not ehough.");
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	this->SumRowTest(path.c_str());
}

// ============================== SGS ======================================
template<typename T>
class BiCGStabSGS : public SolverTest<T> {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps, a.template getPreconditioner<SMM::SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL>());
	}
};

template<typename T>
class BiCGStabSGS_PositiveDefinite : public BiCGStabSGS<T> {

};

template<typename T>
class BiCGStabSGS_Indefinite : public BiCGStabSGS<T> {

};

TYPED_TEST_SUITE(BiCGStabSGS, MyTypes);
TYPED_TEST_SUITE(BiCGStabSGS_PositiveDefinite, MyTypes);
TYPED_TEST_SUITE(BiCGStabSGS_Indefinite, MyTypes);

TYPED_TEST(BiCGStabSGS_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStabSGS_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStabSGS_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(BiCGStabSGS_Indefinite, sherman1_1000_1000_2375) {
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	this->SumRowTest(path.c_str());
}

#endif

#include "doctest/doctest.h"
#include "sparse_matrix_math.h"
#include "test_common.h"
#include <string>
#include <vector>

TEST_CASE_TEMPLATE("BiConjugate Gradient Stabilized method", T, float, double) {
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
	REQUIRE_EQ(SMM::BiCGStab<T>(m, rhs, x, -1, l2Eps<T>()), SMM::SolverStatus::SUCCESS);

	for (const T ri : x) {
		CHECK_EQ(T(1), doctest::Approx(ri).epsilon(infEps<T>()));
	}
}

TEST_CASE_TEMPLATE("Preconditioned BiConjugate Gradient Stabilized method. Symmetric Gauss Seidel Preconditioner", T, float, double) {
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

	using SGSPreconditioner = SMM::CSRMatrix<T>::SGSPreconditioner;
	const SGSPreconditioner& M = m.template getPreconditioner<SMM::SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL>();
	REQUIRE_EQ(SMM::BiCGStab<SGSPreconditioner, T>(m, rhs, x, -1, l2Eps<T>(), M), SMM::SolverStatus::SUCCESS);

	for (const T ri : x) {
		CHECK_EQ(T(1), doctest::Approx(ri).epsilon(infEps<T>()));
	}
}