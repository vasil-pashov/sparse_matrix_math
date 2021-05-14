#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

class CG : public SolverTest {
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps) override {
		return SMM::ConjugateGradient(a, b, x, maxIterations, eps);
	}
};

TEST_F(CG, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(CG, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(CG, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST(PCG, TestApplyIC0) {
	const int size = 5;
	const SMM::real denseRef[size][size] = {
		{10, 0, 0 , 4 , 0},
    	{0 , 9, 0 , 0 , 5},
    	{0 , 0, 12, 0 , 0},
    	{4 , 0, 0 , 15, 7},
    	{0 , 5, 0 , 7 , 8}
	};

	SMM::TripletMatrix triplet(size, size);
	for(int i = 0; i < size; ++i) {
		for(int j = 0; j < size; ++j) {
			if(denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
			}
		}
	}
	SMM::CSRMatrix m;
	m.init(triplet);
	SMM::CSRMatrix::IC0Preconditioner ic0(m);
	int error = ic0.init();
	ASSERT_EQ(error, 0);
	SMM::real rhs[size];
	std::fill_n(rhs, size, 1);
	SMM::real res[size];
	SMM::real resRef[size] = {0.0995763, 0.0646186, 0.0833333, 0.0010593, 0.0836864};
	ic0.apply(rhs, res);
	for(int i = 0; i < size; ++i) {
		EXPECT_NEAR(res[i], resRef[i], 1e-4);
	}
}