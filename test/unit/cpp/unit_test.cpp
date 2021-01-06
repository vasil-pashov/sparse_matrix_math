#include "gtest/gtest.h"
#include "sparse_matrix_math.h"

TEST(TripletMatrixTest, ConstructorRowCol) {
	const int numRows = 5;
	const int numCols = 10;
	SMM::TripletMatrix m(numRows, numCols);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}

TEST(TripletMatrixTest, ConstructorAlloc) {
	const int numRows = 5;
	const int numCols = 10;
	const int numElements = 5;
	SMM::TripletMatrix m(numRows, numCols, numElements);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}
TEST(TripletMatrixTest, AddElementsCount) {
	const int numRows = 10;
	const int numCols = 12;
	SMM::TripletMatrix m(numRows, numCols);
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
	SMM::TripletMatrix m(numRows, numCols, numElements);
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
	SMM::TripletMatrix m(numRows, numCols, numElements);
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

	float dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

// ==========================================================================
// ==================== COMPRESSED SPARSE MATRIX COMMON =====================
// ==========================================================================


template <typename CSMatrix_t>
class CSMatrixCtorTest : public testing::Test {

};

using CSMatrixTypes = ::testing::Types<SMM::CSRMatrix>;
TYPED_TEST_SUITE(CSMatrixCtorTest, CSMatrixTypes);

TYPED_TEST(CSMatrixCtorTest, ConstrcutFromEmptyTriplet) {
	const int numRows = 4;
	const int numCols = 5;
	SMM::TripletMatrix triplet(numRows, numCols);
	TypeParam m(triplet);
	EXPECT_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(m.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
}

TYPED_TEST(CSMatrixCtorTest, ConstructFromTriplet) {
	const int numRows = 10;
	const int numCols = 12;
	const int numElements = 2;
	SMM::TripletMatrix triplet(numRows, numCols, numElements);
	EXPECT_EQ(triplet.getNonZeroCount(), 0);

	triplet.addEntry(1, 1, 1.0f);
	triplet.addEntry(2, 2, 1.0f);
	triplet.addEntry(1, 1, 5.0f);
	triplet.addEntry(3, 5, 10.0f);

	SMM::CSRMatrix csr(triplet);
	EXPECT_EQ(csr.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(csr.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(csr.getNonZeroCount(), triplet.getNonZeroCount());
}

template<typename CSMatrix_t>
using CSMatrixToLinearRowMajor = CSMatrixCtorTest<CSMatrix_t>;

TYPED_TEST_SUITE(CSMatrixToLinearRowMajor, CSMatrixTypes);

TYPED_TEST(CSMatrixToLinearRowMajor, ToLinearDenseRowMajor) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix triplet(numRows, numCols, numElements);
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

	SMM::CSRMatrix csr(triplet);
	EXPECT_EQ(triplet.getNonZeroCount(), csr.getNonZeroCount());

	float dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(csr, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_EQ(dense[i], denseRef[i]);
	}
}

template <typename CSMatrix_t>
class CSMatrixRMultOp : public testing::Test {
protected:
	static const int numRows = 4;
	static const int numCols = 4;
	static const int numElements = 10;
	SMM::TripletMatrix triplet;

	CSMatrixRMultOp() : triplet(numRows, numCols, numElements)
	{
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
		m.init(triplet);
	}
	
	SMM::CSRMatrix m;
};

template<typename CSMatrix_t>
using CSMatrixRMultAdd = CSMatrixRMultOp<CSMatrix_t>;

TYPED_TEST_SUITE(CSMatrixRMultAdd, CSMatrixTypes);

TYPED_TEST(CSMatrixRMultAdd, EmptyMatrix) {
	float mult[numRows] = { 1,2,3,4 };
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix emptyTriplet(4, 4, 10);
	TypeParam emptyMatrix(emptyTriplet);
	emptyMatrix.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddMultZero) {
	float mult[numRows] = {};
	float add[numRows] = {};
	const float resRef[numRows] = {};
	m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddZero) {
	float mult[numRows] = { 1,2,3,4 };
	float add[numRows] = {};
	const float resRef[numRows] = { 14.1, 12.5, 12.4, 8.3 };
	m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, MultZero) {
	float mult[numRows] = {};
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { 5,6,7,8 };
	m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, Basic) {
	float mult[numRows] = { 1,0,3,4 };
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { 19.1, 12.7, 16., 15.5 };
	m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}


template<typename CSMatrix_t>
using CSMatrixRMultSub = CSMatrixRMultOp<CSMatrix_t>;

TYPED_TEST_SUITE(CSMatrixRMultSub, CSMatrixTypes);

TYPED_TEST(CSMatrixRMultSub, EmptyMatrix) {
	float mult[numRows] = { 1,2,3,4 };
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix emptyTriplet(4, 4, 10);
	TypeParam emptyMatrix(emptyTriplet);
	emptyMatrix.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubMultZero) {
	float mult[numRows] = {};
	float add[numRows] = {};
	const float resRef[numRows] = {};
	m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubZero) {
	float mult[numRows] = { 1,2,3,4 };
	float add[numRows] = {};
	const float resRef[numRows] = { -14.1, -12.5, -12.4, -8.3 };
	m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, MultZero) {
	float mult[numRows] = {};
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { 5,6,7,8 };
	m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, Basic) {
	float mult[numRows] = { 1,0,3,4 };
	float add[numRows] = { 5,6,7,8 };
	const float resRef[numRows] = { -9.1, -0.7, -2., 0.5 };
	m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

// ==========================================================================
// ===================== COMPRESSED SPARSE ROW MATRIX  ======================
// ==========================================================================

TEST(CSRMatrixTest, CSRMatrixEmptyConstForwardIterator) {
	SMM::TripletMatrix triplet(10, 10);
	SMM::CSRMatrix csr(triplet);
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

	SMM::TripletMatrix triplet(numRows, numCols);
	triplet.addEntry(4, 4, 4.4f);
	triplet.addEntry(1, 1, 1.1f);
	triplet.addEntry(1, 2, 2.2f);
	triplet.addEntry(1, 3, 3.3f);

	SMM::CSRMatrix csr(triplet);
	SMM::CSRMatrix::ConstIterator it = csr.begin();

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

// =========================================================================
// ============================== BiCGSymmetric ============================
// =========================================================================

TEST(BiCGSymmetric, SmallDenseMatrix) {
	SMM::TripletMatrix triplet(4, 4, 16);
	float dense[16] = { 30.49, 13.95, 9.6, 15.75, 13.95, 18.83, 4.93, 12.91, 9.6, 4.93, 11.89, 0.68, 15.75, 12.91, 0.68, 13.41 };
	for (int row = 0; row < 4; ++row) {
		for (int col = 0; col < 4; ++col) {
			triplet.addEntry(row, col, dense[row * 4 + col]);
		}
	}

	SMM::CSRMatrix csr(triplet);
	SMM::Vector b({1,2,3,4});
	SMM::Vector x(4, 0);

	EXPECT_FALSE(SMM::BiCGSymmetric(csr, b, x));
	float resRef[4] = { -5.57856, -5.62417, 6.40556, 11.9399 };
	for (int i = 0; i < 4; ++i) {
		EXPECT_NEAR(x[i], resRef[i], 1e-4);
	}
}

// =========================================================================
// ============================== FILE LOAD ================================
// =========================================================================

TEST(LoadFile, LoadMatrixMarket) {
	const std::string path = ASSET_PATH + std::string("ex5.mtx");
	SMM::TripletMatrix triplet;
	const SMM::MatrixLoadStatus status = SMM::loadMatrix(path.c_str(), triplet);
	ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
	EXPECT_EQ(triplet.getDenseRowCount(), 27);
	EXPECT_EQ(triplet.getDenseColCount(), 27);
	EXPECT_EQ(triplet.getNonZeroCount(), 279);
}