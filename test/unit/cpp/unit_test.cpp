#include "gtest/gtest.h"
#include "sparse_matrix_math.h"

TEST(TripletMatrixTest, ConstructorRowCol) {
	const int numRows = 5;
	const int numCols = 10;
	SparseMatrix::TripletMatrix m(numRows, numCols);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}

TEST(TripletMatrixTest, ConstructorAlloc) {
	const int numRows = 5;
	const int numCols = 10;
	const int numElements = 5;
	SparseMatrix::TripletMatrix m(numRows, numCols, numElements);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}
TEST(TripletMatrixTest, AddElementsCount) {
	const int numRows = 10;
	const int numCols = 12;
	SparseMatrix::TripletMatrix m(numRows, numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);

	m.addEntry(1, 1, 1.0f);
	EXPECT_EQ(m.getNonZeroCount(), 1);

	m.addEntry(2, 2, 1.0f);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(1, 1, 5.0f);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(3, 5, 10.0f);
	EXPECT_EQ(m.getNonZeroCount(), 3);
}


TEST(TripletMatrixTest, AddElementsAllocPreCount) {
	const int numRows = 10;
	const int numCols = 12;
	const int numElements = 2;
	SparseMatrix::TripletMatrix m(numRows, numCols, numElements);
	EXPECT_EQ(m.getNonZeroCount(), 0);

	m.addEntry(1, 1, 1.0f);
	EXPECT_EQ(m.getNonZeroCount(), 1);

	m.addEntry(2, 2, 1.0f);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(1, 1, 5.0f);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(3, 5, 10.0f);
	EXPECT_EQ(m.getNonZeroCount(), 3);
}

TEST(TripletMatrixTest, ToLinearDenseRowMajor) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SparseMatrix::TripletMatrix m(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	const float denseRef[denseSize] = {
		4.5f, 0.0f, 3.2f, 0.0f,
		3.1f, 2.9f, 0.0f, 0.9f,
		0.0f, 1.7f, 3.0f, 0.0f,
		3.5f, 0.4f, 0.0f, 1.0f
	};
	m.addEntry(0, 0, 3.0f); // Add (0, 0) as sum
	m.addEntry(0, 2, 3.2f);
	m.addEntry(1, 1, 2.9f);
	m.addEntry(0, 0, 1.0f); // Add (0, 0) as sum
	m.addEntry(0, 0, 0.5f); // Add (0, 0) as sum
	m.addEntry(1, 0, 3.1f);
	m.addEntry(1, 3, 0.9f);
	m.addEntry(2, 2, 2.1f); // Add (2,2) as sum
	m.addEntry(2, 1, 2.0f); // Add (2, 1) as sum
	m.addEntry(2, 2, 0.9f); // Add (2,2) as sum
	m.addEntry(2, 1, -0.3f); // Add (2, 1) as sum
	m.addEntry(3, 0, 3.5f);
	m.addEntry(3, 1, 0.4f);
	m.addEntry(3, 3, 1.0f);

	float* dense = static_cast<float*>(calloc(denseSize, sizeof(float)));
	SparseMatrix::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
	free(dense);
}

// ==========================================================================
// ===================== Compressed Sparse Row Matrix =======================
// ==========================================================================

TEST(CSRMatrixTest, ConstrcutFromEmptyTriplet) {
	const int numRows = 4;
	const int numCols = 5;
	SparseMatrix::TripletMatrix triplet(numRows, numCols);
	SparseMatrix::CSRMatrix m(triplet);
	EXPECT_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(m.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
}

TEST(CSRMatrixTest, ConstructFromTriplet) {
	const int numRows = 10;
	const int numCols = 12;
	const int numElements = 2;
	SparseMatrix::TripletMatrix triplet(numRows, numCols, numElements);
	EXPECT_EQ(triplet.getNonZeroCount(), 0);

	triplet.addEntry(1, 1, 1.0f);
	triplet.addEntry(2, 2, 1.0f);
	triplet.addEntry(1, 1, 5.0f);
	triplet.addEntry(3, 5, 10.0f);

	SparseMatrix::CSRMatrix csr(triplet);
	EXPECT_EQ(csr.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(csr.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(csr.getNonZeroCount(), triplet.getNonZeroCount());
}

TEST(CSRMatrixTest, CSRMatrixEmptyConstForwardIterator) {
	SparseMatrix::TripletMatrix triplet(10, 10);
	SparseMatrix::CSRMatrix csr(triplet);
	EXPECT_TRUE(csr.begin() == csr.end());
}

TEST(CSRMatrixTest, CSRMatrixConstForwardIterator) {
	const int numRows = 6;
	const int numCols = 6;
	float denseRef[numRows][numCols] = {};
	denseRef[1][1] = 1.1f;
	denseRef[1][2] = 2.2f;
	denseRef[1][3] = 3.3f;
	denseRef[4][4] = 4.4f;

	SparseMatrix::TripletMatrix triplet(numRows, numCols);
	triplet.addEntry(4, 4, 4.4f);
	triplet.addEntry(1, 1, 1.1f);
	triplet.addEntry(1, 2, 2.2f);
	triplet.addEntry(1, 3, 3.3f);

	SparseMatrix::CSRMatrix csr(triplet);
	SparseMatrix::CSRMatrix::ConstIterator it = csr.begin();

	EXPECT_EQ(it->getRow(), 1);
	EXPECT_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

	++it;
	EXPECT_EQ(it->getRow(), 1);
	EXPECT_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

	++it;
	EXPECT_EQ(it->getRow(), 1);
	EXPECT_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);
	
	++it;
	EXPECT_EQ(it->getRow(), 4);
	EXPECT_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

	++it;
	EXPECT_TRUE(it == csr.end());
}

TEST(CSRMatrixTest, ToLinearDenseRowMajor) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SparseMatrix::TripletMatrix triplet(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	const float denseRef[denseSize] = {
		4.5f, 0.0f, 3.2f, 0.0f,
		3.1f, 2.9f, 0.0f, 0.9f,
		0.0f, 1.7f, 3.0f, 0.0f,
		3.5f, 0.4f, 0.0f, 1.0f
	};
	triplet.addEntry(0, 0, 4.5f);
	triplet.addEntry(0, 2, 3.2f);
	triplet.addEntry(1, 0, 3.1f);
	triplet.addEntry(1, 1, 2.9f);
	triplet.addEntry(1, 3, 0.9f);
	triplet.addEntry(2, 1, 1.7f);
	triplet.addEntry(2, 2, 3.0f);
	triplet.addEntry(3, 0, 3.5f);
	triplet.addEntry(3, 1, 0.4f);
	triplet.addEntry(3, 3, 1.0f);

	SparseMatrix::CSRMatrix csr(triplet);
	EXPECT_TRUE(triplet.getNonZeroCount(), csr.getNonZeroCount());

	float* dense = static_cast<float*>(calloc(denseSize, sizeof(float)));
	SparseMatrix::toLinearDenseRowMajor(csr, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_EQ(dense[i], denseRef[i]);
	}
	free(dense);
}