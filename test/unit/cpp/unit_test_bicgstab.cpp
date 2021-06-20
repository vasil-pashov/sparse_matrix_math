#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

template<typename T>
class BiCGStab : public SolverTest<T> {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, T* b, T* x, int maxIterations, T eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps);
	}
};

template<typename T>
class BiCGStab_PositiveDefinite : public BiCGStab<T> {

};

template<typename T>
class BiCGStab_Indefinite : public BiCGStab<T> {

};


using MyTypes = ::testing::Types<SMM::real>;
TYPED_TEST_SUITE(BiCGStab, MyTypes);
TYPED_TEST_SUITE(BiCGStab_PositiveDefinite, MyTypes);
TYPED_TEST_SUITE(BiCGStab_Indefinite, MyTypes);

TYPED_TEST(BiCGStab, SmallDenseMatrix) {
	GTEST_SKIP_("The numerical instability of the vector is clearly seen here. Using fma helps a little, but not ehough.");
	SMM::TripletMatrix<TypeParam> triplet(4, 4, 16);
	SMM::real dense[16] = { 30.49, 13.95, 9.6, 15.75, 13.95, 18.83, 4.93, 12.91, 9.6, 4.93, 11.89, 0.68, 15.75, 12.91, 0.68, 13.41 };
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			triplet.addEntry(row, col, dense[row * 4 + col]);
		}
	}

	SMM::CSRMatrix csr(triplet);
	SMM::Vector<SMM::real> b({1,2,3,4});
	SMM::Vector<SMM::real> x(4, 0);

	EXPECT_EQ(this->solve(csr, b, x, -1, L2Epsilon()), SMM::SolverStatus::SUCCESS);
	SMM::real resRef[4] = { -5.57856, -5.62417, 6.40556, 11.9399 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(x[i], resRef[i], MaxInfEpsilon());
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
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, T* b, T* x, int maxIterations, T eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps, a.getPreconditioner<SMM::SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL>());
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