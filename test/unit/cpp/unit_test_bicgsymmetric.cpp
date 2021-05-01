#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

class BiCGSymmetric : public SolverTest {
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps) override {
		return SMM::BiCGSymmetric(a, b, x, maxIterations, eps);
	}
};

class BiCGSymmetric_PositiveDefinite : public BiCGSymmetric {

};

class BiCGSymmetric_Indefinite : public BiCGSymmetric {

};

TEST_F(BiCGSymmetric, SmallDenseMatrix) {
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

	EXPECT_EQ(SMM::BiCGSymmetric(csr, b, x, -1, L2Epsilon()), SMM::SolverStatus::SUCCESS);
	SMM::real resRef[4] = { -5.57856, -5.62417, 6.40556, 11.9399 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(x[i], resRef[i], MaxInfEpsilon());
	}
}

TEST_F(BiCGSymmetric_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSymmetric_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSymmetric_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGSymmetric_Indefinite, sherman1_1000_1000_2375) {
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	SumRowTest(path.c_str());
}