#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

class BiCGSquared : public SolverTest {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps) override {
		return SMM::BiCGSquared(a, b, x, maxIterations, eps);
	}
};

class BiCGSquared_PositiveDefinite : public BiCGSquared {

};

class BiCGSquared_Indefinite : public BiCGSquared {

};

TEST_F(BiCGSquared, SmallDenseMatrix) {
	GTEST_SKIP_("The numerical instability of the vector is clearly seen here. Using fma helps a little, but not ehough.");
	SMM::TripletMatrix triplet(4, 4, 16);
	SMM::real dense[16] = { 30.49, 13.95, 9.6, 15.75, 13.95, 18.83, 4.93, 12.91, 9.6, 4.93, 11.89, 0.68, 15.75, 12.91, 0.68, 13.41 };
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			triplet.addEntry(row, col, dense[row * 4 + col]);
		}
	}

	SMM::CSRMatrix csr(triplet);
	SMM::Vector b({1,2,3,4});
	SMM::Vector x(4, 0);

	EXPECT_EQ(solve(csr, b, x, -1, L2Epsilon()), SMM::SolverStatus::SUCCESS);
	SMM::real resRef[4] = { -5.57856, -5.62417, 6.40556, 11.9399 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(x[i], resRef[i], MaxInfEpsilon());
	}
}

TEST_F(BiCGSquared_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSquared_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSquared_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSquared_PositiveDefinite, sherman1_1000_1000_2375) {
	GTEST_SKIP_("The numerical instability of the method is clearly seen here. Using fma helps a little, but not ehough.");
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	SumRowTest(path.c_str());
}