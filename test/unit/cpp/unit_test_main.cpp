#include "gtest/gtest.h"
#include "test_common.h"
#include <vector>
#include <utility>
namespace SMM {
	template<typename A, typename B>
	testing::AssertionResult NearMatrix(const A& lhs, const B& rhs, const real eps) {
		if (lhs.getNonZeroCount() != rhs.getNonZeroCount()) {
			return testing::AssertionFailure() << "Matrices have different count of non zero elements. lhs has: " << lhs.getNonZeroCount() << " rhs has: " << rhs.getNonZeroCount();
		}
		if (lhs.getDenseRowCount() != rhs.getDenseRowCount()) {
			return testing::AssertionFailure() << "Matrices have different count of rows. lhs has: " << lhs.getDenseRowCount() << " rhs has: " << rhs.getDenseRowCount();

		}
		if (lhs.getDenseColCount() != rhs.getDenseColCount()) {
			return testing::AssertionFailure() << "Matrices have different count of columns. lhs has: " << lhs.getDenseColCount() << " rhs has: " << rhs.getDenseColCount();
		}
		const size_t sz = size_t(lhs.getDenseColCount()) * lhs.getDenseRowCount();
		std::vector<real> adense(sz, 0);
		std::vector<real> bdense(sz, 0);
		SMM::toLinearDenseRowMajor(lhs, adense.data());
		SMM::toLinearDenseRowMajor(rhs, bdense.data());
		for (int row = 0; row < lhs.getDenseRowCount(); ++row) {
			for (int col = 0; col < rhs.getDenseColCount(); ++col) {
				const int64_t index = row * lhs.getDenseColCount() + col;
				if (std::abs(adense[index] - bdense[index]) > eps) {
					return testing::AssertionFailure() << "Matrices have differing elements. lhs[" << row << "][" << col << "] = "
						<< adense[index] << " != rhs[" << row << "][" << col << "] = " << bdense[index] << "."
						<< "Epsilon is: " << eps << " while difference is: " << std::abs(adense[index] - bdense[index]);
				}
			}
		}
		return testing::AssertionSuccess();
	}
}

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
	const SMM::real denseRef[denseSize] = {
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

	SMM::real dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

TEST(TripletMatrixTest, DirectAccessors) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix m(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	SMM::real denseRef[denseSize] = {
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

	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			EXPECT_EQ(m.getValue(i, j), denseRef[i * numCols + j]);
		}
	}

	// Update non existing element
	EXPECT_FALSE(m.updateEntry(0, 1, 2));

	// Update existing element
	EXPECT_TRUE(m.updateEntry(1, 3, 200));

	EXPECT_EQ(m.getValue(1, 3), 200);

	denseRef[1 * numCols + 3] = 200;

	SMM::real dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

// ==========================================================================
// ==================== COMPRESSED SPARSE MATRIX COMMON =====================
// ==========================================================================

TEST(CSRMatrix, DirectAccessors) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix triplet(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	SMM::real denseRef[denseSize] = {
		4.5f, 0.0f, 3.2f, 0.0f,
		3.1f, 2.9f, 0.0f, 0.9f,
		0.0f, 1.7f, 3.0f, 0.0f,
		3.5f, 0.4f, 0.0f, 1.0f
	};
	triplet.addEntry(0, 0, 3.0f); // Add (0, 0) as sum
	triplet.addEntry(0, 2, 3.2f);
	triplet.addEntry(1, 1, 2.9f);
	triplet.addEntry(0, 0, 1.0f); // Add (0, 0) as sum
	triplet.addEntry(0, 0, 0.5f); // Add (0, 0) as sum
	triplet.addEntry(1, 0, 3.1f);
	triplet.addEntry(1, 3, 0.9f);
	triplet.addEntry(2, 2, 2.1f); // Add (2,2) as sum
	triplet.addEntry(2, 1, 2.0f); // Add (2, 1) as sum
	triplet.addEntry(2, 2, 0.9f); // Add (2,2) as sum
	triplet.addEntry(2, 1, -0.3f); // Add (2, 1) as sum
	triplet.addEntry(3, 0, 3.5f);
	triplet.addEntry(3, 1, 0.4f);
	triplet.addEntry(3, 3, 1.0f);

	SMM::CSRMatrix m;
	m.init(triplet);

	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			EXPECT_EQ(m.getValue(i, j), denseRef[i * numCols + j]);
		}
	}

	// Update non existing element
	EXPECT_FALSE(m.updateEntry(0, 1, 2));

	// Update existing element
	EXPECT_TRUE(m.updateEntry(1, 3, 200));

	EXPECT_EQ(m.getValue(1, 3), 200);

	denseRef[1 * numCols + 3] = 200;

	SMM::real dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

TEST(CSRMatrix, RowIterators) {
	const int numRows = 8;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix triplet(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	const SMM::real denseRef[denseSize] = {
		0.0f, 0.0f, 0.0f, 0.0f,
		4.5f, 0.0f, 3.2f, 0.0f,
		3.1f, 2.9f, 0.0f, 0.9f,
		0.0f, 1.7f, 3.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f,
		0.0f, 0.0f, 0.0f, 0.0f,
		3.5f, 0.4f, 0.0f, 1.0f,
		0.0f, 0.0f, 0.0f, 0.0f
	};
	int numElementsInRow[]= {2,3,2,0,3};

	triplet.addEntry(0, 0, 4.5f);
	triplet.addEntry(0, 2, 3.2f);
	triplet.addEntry(1, 0, 3.1f);
	triplet.addEntry(1, 1, 2.9f);
	triplet.addEntry(1, 3, 0.9f);
	triplet.addEntry(2, 1, 1.7f);
	triplet.addEntry(2, 2, 3.0f);
	triplet.addEntry(4, 0, 3.5f);
	triplet.addEntry(4, 1, 0.4f);
	triplet.addEntry(4, 3, 1.0f);

	SMM::CSRMatrix m;
	m.init(triplet);

	for(int i = 0; i < numRows; ++i) {
		int numElements = 0;
		SMM::CSRMatrix::ConstRowIterator current = m.rowBegin(i);
		SMM::CSRMatrix::ConstRowIterator rowEnd = m.rowEnd(i);
		while(current != rowEnd) {
			EXPECT_EQ(current->getValue(), denseRef[current->getRow() * numCols + current->getCol()]);
			numElements++;
			++current;
		}
		EXPECT_EQ(numElements, numElementsInRow[i]);
	}

}


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
	const SMM::real denseRef[denseSize] = {
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

	SMM::real dense[denseSize] = {};
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
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,2,3,4 };
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix emptyTriplet(4, 4, 10);
	TypeParam emptyMatrix(emptyTriplet);
	emptyMatrix.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddMultZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	SMM::real mult[numRows] = {};
	SMM::real add[numRows] = {};
	const SMM::real resRef[numRows] = {};
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,2,3,4 };
	SMM::real add[numRows] = {};
	const SMM::real resRef[numRows] = { 14.1, 12.5, 12.4, 8.3 };
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, MultZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	SMM::real mult[numRows] = {};
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { 5,6,7,8 };
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, Basic) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,0,3,4 };
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { 19.1, 12.7, 16., 15.5 };
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}


template<typename CSMatrix_t>
using CSMatrixRMultSub = CSMatrixRMultOp<CSMatrix_t>;

TYPED_TEST_SUITE(CSMatrixRMultSub, CSMatrixTypes);

TYPED_TEST(CSMatrixRMultSub, EmptyMatrix) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,2,3,4 };
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix emptyTriplet(4, 4, 10);
	TypeParam emptyMatrix(emptyTriplet);
	emptyMatrix.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubMultZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	SMM::real mult[numRows] = {};
	SMM::real add[numRows] = {};
	const SMM::real resRef[numRows] = {};
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,2,3,4 };
	SMM::real add[numRows] = {};
	const SMM::real resRef[numRows] = { -14.1, -12.5, -12.4, -8.3 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, MultZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	SMM::real mult[numRows] = {};
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { 5,6,7,8 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, Basic) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	SMM::real mult[numRows] = { 1,0,3,4 };
	SMM::real add[numRows] = { 5,6,7,8 };
	const SMM::real resRef[numRows] = { -9.1, -0.7, -2., 0.5 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
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
	SMM::real denseRef[numRows][numCols] = {};
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
// ======================= Arithmetic operations ===========================
// =========================================================================

TEST(CSRArithmetic, MultiplyByScalar) {
	const int numRows = 6;
	const int numCols = 10;

	SMM::real denseRef[numRows][numCols] = {};
	denseRef[2][8] = 28.41637;
	denseRef[2][2] = 31.52779;
	denseRef[1][7] = -237.59453;
	denseRef[5][3] = 273.3937;
	denseRef[0][3] = -471.11824;

	SMM::TripletMatrix triplet(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
			}
		}
	}

	SMM::CSRMatrix csr;
	csr.init(triplet);

	const SMM::real scalar = -478.53439;

	csr *= scalar;

	for(const auto& el : csr) {
		EXPECT_EQ(el.getValue(), scalar * denseRef[el.getRow()][el.getCol()]);
	}
}


TEST(CSRArithmetic, InplaceAdd) {
	const int numRows = 6;
	const int numCols = 10;

	SMM::real denseRef1[numRows][numCols] = {};
	denseRef1[2][8] = 28.41637;
	denseRef1[2][2] = 31.52779;
	denseRef1[1][7] = -237.59453;
	denseRef1[5][3] = 273.3937;
	denseRef1[0][3] = -471.11824;

	SMM::real denseRef2[numRows][numCols] = {};
	denseRef1[2][8] = 558.57004;
	denseRef1[2][2] = 53.47841;
	denseRef1[1][7] = 621.94377;
	denseRef1[5][3] = 237.2853;
	denseRef1[0][3] = -449.43152;

	SMM::TripletMatrix triplet1(numRows, numCols);
	SMM::TripletMatrix triplet2(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef1[i][j] != 0) {
				triplet1.addEntry(i, j, denseRef1[i][j]);
				triplet2.addEntry(i, j, denseRef2[i][j]);
			}
		}
	}

	SMM::CSRMatrix csr1;
	csr1.init(triplet1);

	SMM::CSRMatrix csr2;
	csr2.init(triplet2);

	csr1.inplaceAdd(csr2);

	for(const auto& el : csr1) {
		EXPECT_EQ(el.getValue(), denseRef1[el.getRow()][el.getCol()] + denseRef2[el.getRow()][el.getCol()]);
	}
}

TEST(CSRArithmetic, InplaceSubtract) {
	const int numRows = 6;
	const int numCols = 10;

	SMM::real denseRef1[numRows][numCols] = {};
	denseRef1[2][8] = 28.41637;
	denseRef1[2][2] = 31.52779;
	denseRef1[1][7] = -237.59453;
	denseRef1[5][3] = 273.3937;
	denseRef1[0][3] = -471.11824;

	SMM::real denseRef2[numRows][numCols] = {};
	denseRef1[2][8] = 558.57004;
	denseRef1[2][2] = 53.47841;
	denseRef1[1][7] = 621.94377;
	denseRef1[5][3] = 237.2853;
	denseRef1[0][3] = -449.43152;

	SMM::TripletMatrix triplet1(numRows, numCols);
	SMM::TripletMatrix triplet2(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef1[i][j] != 0) {
				triplet1.addEntry(i, j, denseRef1[i][j]);
				triplet2.addEntry(i, j, denseRef2[i][j]);
			}
		}
	}

	SMM::CSRMatrix csr1;
	csr1.init(triplet1);

	SMM::CSRMatrix csr2;
	csr2.init(triplet2);

	csr1.inplaceSubtract(csr2);

	for(const auto& el : csr1) {
		EXPECT_EQ(el.getValue(), denseRef1[el.getRow()][el.getCol()] - denseRef2[el.getRow()][el.getCol()]);
	}
}

// =========================================================================
// ============================== FILE LOAD ================================
// =========================================================================

TEST(LoadFile, LoadMatrixMarket) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SMM::TripletMatrix triplet;
	const SMM::MatrixLoadStatus status = SMM::loadMatrix(path.c_str(), triplet);
	ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
	EXPECT_EQ(triplet.getDenseRowCount(), 48);
	EXPECT_EQ(triplet.getDenseColCount(), 48);
	EXPECT_EQ(triplet.getNonZeroCount(), 306);
}

// =========================================================================
// ============================== FILE SAVE ================================
// =========================================================================

TEST(SaveDense, CSRSMMDT) {
	struct FileRemoveRAII {
		FileRemoveRAII(const std::string& name) :
			name(name)
		{ }
		~FileRemoveRAII() {
			std::remove(name.c_str());
		}
		const char* getName() {
			return name.c_str();
		}
	private:
		std::string name;
	};
	SMM::TripletMatrix triplet(10, 10);
	triplet.addEntry(0, 0, 234.5324);
	triplet.addEntry(3, 2, 2.4);
	triplet.addEntry(5, 2, 1);
	triplet.addEntry(5, 3, 2);
	triplet.addEntry(5, 4, 3);
	triplet.addEntry(5, 6, 4);
	triplet.addEntry(6, 1, 3.4);
	for (int i = 0; i < 10; ++i) triplet.addEntry(9, i, 1);
	SMM::CSRMatrix csr(triplet);
	FileRemoveRAII file(ASSET_PATH + std::string("__test.smmdt"));
	SMM::saveDenseText(file.getName(), csr);

	SMM::TripletMatrix tripletRead;
	const SMM::MatrixLoadStatus status = SMM::loadMatrix(file.getName(), tripletRead);
	ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
	EXPECT_TRUE(NearMatrix(tripletRead, csr, 1e-6));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}