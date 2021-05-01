#include "gtest/gtest.h"
#include "test_common.h"
// =========================================================================
// ============================== BiCGSymmetric ============================
// =========================================================================
class SolverTest : public ::testing::Test {
protected:
	virtual SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps)  = 0;
	void SumRowTest(const char* path) {
		SMM::TripletMatrix triplet;
		const SMM::MatrixLoadStatus status = SMM::loadMatrix(path, triplet);
		ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
		SMM::Vector rhs(triplet.getDenseRowCount(), 0.0f);
		// Prepare a vector of right hand sides.
		// Use the sum of each row, as this way the result is known: it's all ones
		for (const auto& el : triplet) {
			rhs[el.getRow()] += el.getValue();
		}


		SMM::CSRMatrix m(triplet);
		SMM::Vector x(m.getDenseRowCount(), 0.0f);
		SMM::SolverStatus solverStatus = solve(m, rhs, x, -1, L2Epsilon());
		EXPECT_EQ(solverStatus, SMM::SolverStatus::SUCCESS);
		SMM::real error = 0.0f;
		for (const SMM::real el : x) {
			error = std::max(abs(1 - el), error);
		}
		EXPECT_LE(error, MaxInfEpsilon());
	}
};

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

// =========================================================================
// ============================== BiCGSquared ==============================
// =========================================================================

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

// =========================================================================
// =========================== BiCGStabilized ==============================
// =========================================================================

class BiCGStab : public SolverTest {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps);
	}
};

class BiCGStab_PositiveDefinite : public BiCGStab {

};

class BiCGStab_Indefinite : public BiCGStab {

};

TEST_F(BiCGStab, SmallDenseMatrix) {
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

TEST_F(BiCGStab_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStab_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStab_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStab_Indefinite, sherman1_1000_1000_2375) {
	GTEST_SKIP_("The numerical instability of the method is clearly seen here. Using fma helps a little, but not ehough.");
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	SumRowTest(path.c_str());
}

// ============================== SGS ======================================
class BiCGStabSGS : public SolverTest {
protected:
	SMM::SolverStatus solve(const SMM::CSRMatrix& a, SMM::real* b, SMM::real* x, int maxIterations, SMM::real eps) override {
		return SMM::BiCGStab(a, b, x, maxIterations, eps, a.getPreconditioner<SMM::SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL>());
	}
};

class BiCGStabSGS_PositiveDefinite : public BiCGStabSGS {

};

class BiCGStabSGS_Indefinite : public BiCGStabSGS {

};

TEST_F(BiCGStabSGS_PositiveDefinite, mesh1e1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStabSGS_PositiveDefinite, mesh1em1_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em1_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStabSGS_PositiveDefinite, mesh1em6_structural_48_48_177) {
	const std::string path = ASSET_PATH + std::string("mesh1em6_structural_48_48_177.mtx");
	SumRowTest(path.c_str());
}

TEST_F(BiCGStabSGS_Indefinite, sherman1_1000_1000_2375) {
	const std::string path = ASSET_PATH + std::string("sherman1_1000_1000_2375.mtx");
	SumRowTest(path.c_str());
}