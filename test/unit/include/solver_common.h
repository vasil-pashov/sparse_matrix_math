//**************************************************************************
// This file contains functions and classes shared by different solver tests
//**************************************************************************
#pragma once
#include "sparse_matrix_math.h"
template<typename T>
class SolverTest : public ::testing::Test {
protected:
	virtual SMM::SolverStatus solve(const SMM::CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps)  = 0;
	void SumRowTest(const char* path) {
		SMM::TripletMatrix<T> triplet;
		const SMM::MatrixLoadStatus status = SMM::loadMatrix(path, triplet);
		ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
		SMM::Vector<T> rhs(triplet.getDenseRowCount(), 0.0);
		// Prepare a vector of right hand sides.
		// Use the sum of each row, as this way the result is known: it's all ones
		for (const auto& el : triplet) {
			rhs[el.getRow()] += el.getValue();
		}


		SMM::CSRMatrix<T> m(triplet);
		SMM::Vector<T> x(m.getDenseRowCount(), 0.0);
		SMM::SolverStatus solverStatus = solve(m, rhs, x, -1, L2Epsilon<T>());
		EXPECT_EQ(solverStatus, SMM::SolverStatus::SUCCESS);
		T error = 0.0;
		for (const T el : x) {
			error = std::max(abs(1 - el), error);
		}
		EXPECT_LE(error, MaxInfEpsilon<T>());
	}
};