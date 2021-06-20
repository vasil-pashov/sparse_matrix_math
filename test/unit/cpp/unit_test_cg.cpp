#include "gtest/gtest.h"
#include "test_common.h"
#include "solver_common.h"

template<typename T>
class CG : public SolverTest<T> {
	SMM::SolverStatus solve(const SMM::CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps) override {
		return SMM::ConjugateGradient(a, b, x, x, maxIterations, eps);
	}
};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CG, MyTypes);

TYPED_TEST(CG, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(CG, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(CG, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TEST(IC0, TestApplyIC0) {
	const int size = 5;
	float denseRef[size][size] = {
		{10, 0, 0 , 4 , 0},
    	{0 , 9, 0 , 0 , 5},
    	{0 , 0, 12, 0 , 0},
    	{4 , 0, 0 , 15, 7},
    	{0 , 5, 0 , 7 , 8}
	};

	SMM::TripletMatrix<float> triplet(size, size);
	for(int i = 0; i < size; ++i) {
		for(int j = 0; j < size; ++j) {
			if(denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
			}
		}
	}
	SMM::CSRMatrix<float> m;
	m.init(triplet);
	SMM::CSRMatrix<float>::IC0Preconditioner ic0(m);
	int error = ic0.init();
	ASSERT_EQ(error, 0);
	float rhs[size];
	std::fill_n(rhs, size, 1);
	float res[size];
	float resRef[size] = {0.0995763, 0.0646186, 0.0833333, 0.0010593, 0.0836864};
	ic0.apply(rhs, res);
	for(int i = 0; i < size; ++i) {
		EXPECT_NEAR(res[i], resRef[i], 1e-4);
	}
}

TYPED_TEST_SUITE(PCG, MyTypes);

template<typename T>
class PCG : public SolverTest<T> {
	SMM::SolverStatus solve(const SMM::CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps) override {
		typename SMM::CSRMatrix<T>::IC0Preconditioner M(a);
		const int error = M.init();
		EXPECT_EQ(error, 0);
		return SMM::ConjugateGradient(a, b, x, x, maxIterations, eps, M);
	}
};

TYPED_TEST(PCG, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(PCG, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}

TYPED_TEST(PCG, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	this->SumRowTest(path.c_str());
}