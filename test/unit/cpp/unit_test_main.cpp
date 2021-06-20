#include "gtest/gtest.h"
#include "test_common.h"
#include <vector>
#include <utility>
namespace SMM {
	template<typename A, typename B, typename T = std::enable_if_t<std::is_same_v<typename A::value_type, typename B::value_type>, typename A::value_type>>
	testing::AssertionResult NearMatrix(const A& lhs, const B& rhs, const T eps) {
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
		std::vector<typename A::value_type> adense(sz, 0);
		std::vector<typename A::value_type> bdense(sz, 0);
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

template <typename T>
class TripletMatrixTest : public testing::Test {};

using MyTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(TripletMatrixTest, MyTypes);


TYPED_TEST(TripletMatrixTest, ConstructorRowCol) {
	const int numRows = 5;
	const int numCols = 10;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}

TYPED_TEST(TripletMatrixTest, ConstructorAlloc) {
	const int numRows = 5;
	const int numCols = 10;
	const int numElements = 5;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols, numElements);
	EXPECT_EQ(m.getDenseRowCount(), numRows);
	EXPECT_EQ(m.getDenseColCount(), numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);
}
TYPED_TEST(TripletMatrixTest, AddElementsCount) {
	const int numRows = 10;
	const int numCols = 12;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols);
	EXPECT_EQ(m.getNonZeroCount(), 0);

	m.addEntry(1, 1, 1.0);
	EXPECT_EQ(m.getNonZeroCount(), 1);

	m.addEntry(2, 2, 1.0);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(1, 1, 5.0);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(3, 5, 10.0);
	EXPECT_EQ(m.getNonZeroCount(), 3);
}


TYPED_TEST(TripletMatrixTest, AddElementsAllocPreCount) {
	const int numRows = 10;
	const int numCols = 12;
	const int numElements = 2;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols, numElements);
	EXPECT_EQ(m.getNonZeroCount(), 0);

	m.addEntry(1, 1, 1.0);
	EXPECT_EQ(m.getNonZeroCount(), 1);

	m.addEntry(2, 2, 1.0);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(1, 1, 5.0);
	EXPECT_EQ(m.getNonZeroCount(), 2);

	m.addEntry(3, 5, 10.0);
	EXPECT_EQ(m.getNonZeroCount(), 3);
}

TYPED_TEST(TripletMatrixTest, ToLinearDenseRowMajor) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	const TypeParam denseRef[denseSize] = {
		4.5, 0.0, 3.2, 0.0,
		3.1, 2.9, 0.0, 0.9,
		0.0, 1.7, 3.0, 0.0,
		3.5, 0.4, 0.0, 1.0
	};
	m.addEntry(0, 0, 3.0); // Add (0, 0) as sum
	m.addEntry(0, 2, 3.2);
	m.addEntry(1, 1, 2.9);
	m.addEntry(0, 0, 1.0); // Add (0, 0) as sum
	m.addEntry(0, 0, 0.5); // Add (0, 0) as sum
	m.addEntry(1, 0, 3.1);
	m.addEntry(1, 3, 0.9);
	m.addEntry(2, 2, 2.1); // Add (2,2) as sum
	m.addEntry(2, 1, 2.0); // Add (2, 1) as sum
	m.addEntry(2, 2, 0.9); // Add (2,2) as sum
	m.addEntry(2, 1, -0.3); // Add (2, 1) as sum
	m.addEntry(3, 0, 3.5);
	m.addEntry(3, 1, 0.4);
	m.addEntry(3, 3, 1.0);

	TypeParam dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

TYPED_TEST(TripletMatrixTest, DirectAccessors) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix<TypeParam> m(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	TypeParam denseRef[denseSize] = {
		4.5, 0.0, 3.2, 0.0,
		3.1, 2.9, 0.0, 0.9,
		0.0, 1.7, 3.0, 0.0,
		3.5, 0.4, 0.0, 1.0
	};
	m.addEntry(0, 0, 3.0); // Add (0, 0) as sum
	m.addEntry(0, 2, 3.2);
	m.addEntry(1, 1, 2.9);
	m.addEntry(0, 0, 1.0); // Add (0, 0) as sum
	m.addEntry(0, 0, 0.5); // Add (0, 0) as sum
	m.addEntry(1, 0, 3.1);
	m.addEntry(1, 3, 0.9);
	m.addEntry(2, 2, 2.1); // Add (2,2) as sum
	m.addEntry(2, 1, 2.0); // Add (2, 1) as sum
	m.addEntry(2, 2, 0.9); // Add (2,2) as sum
	m.addEntry(2, 1, -0.3); // Add (2, 1) as sum
	m.addEntry(3, 0, 3.5);
	m.addEntry(3, 1, 0.4);
	m.addEntry(3, 3, 1.0);

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

	TypeParam dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

// ==========================================================================
// ==================== COMPRESSED SPARSE MATRIX COMMON =====================
// ==========================================================================

template <typename T>
class CSRMatrixTest : public testing::Test {};

TYPED_TEST_SUITE(CSRMatrixTest, MyTypes);

TYPED_TEST(CSRMatrixTest, DirectAccessors) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	TypeParam denseRef[denseSize] = {
		4.5, 0.0, 3.2, 0.0,
		3.1, 2.9, 0.0, 0.9,
		0.0, 1.7, 3.0, 0.0,
		3.5, 0.4, 0.0, 1.0
	};
	triplet.addEntry(0, 0, 3.0); // Add (0, 0) as sum
	triplet.addEntry(0, 2, 3.2);
	triplet.addEntry(1, 1, 2.9);
	triplet.addEntry(0, 0, 1.0); // Add (0, 0) as sum
	triplet.addEntry(0, 0, 0.5); // Add (0, 0) as sum
	triplet.addEntry(1, 0, 3.1);
	triplet.addEntry(1, 3, 0.9);
	triplet.addEntry(2, 2, 2.1); // Add (2,2) as sum
	triplet.addEntry(2, 1, 2.0); // Add (2, 1) as sum
	triplet.addEntry(2, 2, 0.9); // Add (2,2) as sum
	triplet.addEntry(2, 1, -0.3); // Add (2, 1) as sum
	triplet.addEntry(3, 0, 3.5);
	triplet.addEntry(3, 1, 0.4);
	triplet.addEntry(3, 3, 1.0);

	SMM::CSRMatrix<TypeParam> m;
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

	TypeParam dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(m, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_NEAR(dense[i], denseRef[i], 10e-6);
	}
}

TYPED_TEST(CSRMatrixTest, RowIterators) {
	const int numRows = 9;
	const int numCols = 4;
	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
	const int denseSize = numRows * numCols;
	const TypeParam denseRef[numRows][numCols] = {
		{0.0, 0.0, 0.0, 0.0},
		{4.5, 0.0, 3.2, 0.0},
		{3.1, 2.9, 0.0, 0.9},
		{0.0, 1.7, 3.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0},
		{3.5, 0.4, 0.0, 1.0},
		{0.0, 0.0, 0.0, 0.0},
		{0.0, 0.0, 0.0, 0.0}
	};
	int numElementsInRow[numRows] = {};
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
				numElementsInRow[i]++;
			}
		}
	}

	SMM::CSRMatrix<TypeParam> m;
	m.init(triplet);

	for(int i = 0; i < numRows; ++i) {
		int numElements = 0;
		typename SMM::CSRMatrix<TypeParam>::ConstRowIterator current = m.rowBegin(i);
		typename SMM::CSRMatrix<TypeParam>::ConstRowIterator rowEnd = m.rowEnd(i);
		while(current != rowEnd) {
			EXPECT_EQ(current->getValue(), denseRef[current->getRow()][current->getCol()]);
			numElements++;
			++current;
		}
		EXPECT_EQ(numElements, numElementsInRow[i]);
	}

	{
		TypeParam current = 0.0;
		for(int i = 0; i < numRows; ++i) {
			typename SMM::CSRMatrix<TypeParam>::RowIterator row = m.rowBegin(i);
			for(int j = 0; j < numCols && row != m.rowEnd(i); ++j) {
				row->setValue(current++);
				++row;
			}
		}

		current = 0;
		for(const auto& el : m) {
			EXPECT_EQ(el.getValue(), current++);
		}
	}

}


template <typename CSMatrix_t>
class CSMatrixCtorTest : public testing::Test {

};

TYPED_TEST_SUITE(CSMatrixCtorTest, MyTypes);

TYPED_TEST(CSMatrixCtorTest, ConstrcutFromEmptyTriplet) {
	const int numRows = 4;
	const int numCols = 5;
	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
	SMM::CSRMatrix m(triplet);
	EXPECT_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(m.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
}

TYPED_TEST(CSMatrixCtorTest, ConstructFromTriplet) {
	const int numRows = 10;
	const int numCols = 12;
	const int numElements = 2;
	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols, numElements);
	EXPECT_EQ(triplet.getNonZeroCount(), 0);

	triplet.addEntry(1, 1, 1.0);
	triplet.addEntry(2, 2, 1.0);
	triplet.addEntry(1, 1, 5.0);
	triplet.addEntry(3, 5, 10.0);

	SMM::CSRMatrix csr(triplet);
	EXPECT_EQ(csr.getDenseRowCount(), triplet.getDenseRowCount());
	EXPECT_EQ(csr.getDenseColCount(), triplet.getDenseColCount());
	EXPECT_EQ(csr.getNonZeroCount(), triplet.getNonZeroCount());
}

template<typename CSMatrix_t>
using CSMatrixToLinearRowMajor = CSMatrixCtorTest<CSMatrix_t>;

TYPED_TEST_SUITE(CSMatrixToLinearRowMajor, MyTypes);

TYPED_TEST(CSMatrixToLinearRowMajor, ToLinearDenseRowMajor) {
	const int numRows = 4;
	const int numCols = 4;
	const int numElements = 10;
	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols, numElements);
	const int denseSize = numRows * numCols;
	const TypeParam denseRef[denseSize] = {
		4.5, 0.0, 3.2, 0.0,
		3.1, 2.9, 0.0, 0.9,
		0.0, 1.7, 3.0, 0.0,
		3.5, 0.4, 0.0, 1.0
	};
	triplet.addEntry(0, 0, 4.5);
	triplet.addEntry(0, 2, 3.2);
	triplet.addEntry(1, 0, 3.1);
	triplet.addEntry(1, 1, 2.9);
	triplet.addEntry(1, 3, 0.9);
	triplet.addEntry(2, 1, 1.7);
	triplet.addEntry(2, 2, 3.0);
	triplet.addEntry(3, 0, 3.5);
	triplet.addEntry(3, 1, 0.4);
	triplet.addEntry(3, 3, 1.0);

	SMM::CSRMatrix csr(triplet);
	EXPECT_EQ(triplet.getNonZeroCount(), csr.getNonZeroCount());

	TypeParam dense[denseSize] = {};
	SMM::toLinearDenseRowMajor(csr, dense);
	for (int i = 0; i < denseSize; ++i) {
		EXPECT_EQ(dense[i], denseRef[i]);
	}
}

template <typename T>
class CSMatrixRMultOp : public testing::Test {
protected:
	static const int numRows = 5;
	static const int numCols = 4;
	static const int numElements = 10;
	SMM::TripletMatrix<T> triplet;

	CSMatrixRMultOp() : triplet(numRows, numCols, numElements)
	{
		triplet.addEntry(0, 0, 4.5);
		triplet.addEntry(0, 2, 3.2);
		triplet.addEntry(1, 0, 3.1);
		triplet.addEntry(1, 1, 2.9);
		triplet.addEntry(1, 3, 0.9);
		triplet.addEntry(2, 1, 1.7);
		triplet.addEntry(2, 2, 3.0);
		triplet.addEntry(3, 0, 3.5);
		triplet.addEntry(3, 1, 0.4);
		triplet.addEntry(3, 3, 1.0);
		m.init(triplet);
	}
	
	SMM::CSRMatrix<T> m;
};

template<typename T>
using CSMatrixRMultAdd = CSMatrixRMultOp<T>;
TYPED_TEST_SUITE(CSMatrixRMultAdd, MyTypes);

TYPED_TEST(CSMatrixRMultAdd, EmptyMatrix) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,2,3,4 };
	TypeParam add[numRows] = { 5,6,7,8 };
	TypeParam resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix<TypeParam> emptyTriplet(4, 4, 10);
	SMM::CSRMatrix emptyMatrix(emptyTriplet);
	emptyMatrix.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddMultZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	TypeParam mult[numRows] = {};
	TypeParam add[numRows] = {};
	const TypeParam resRef[numRows] = {};
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, AddZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,2,3,4 };
	TypeParam add[numRows] = {};
	const TypeParam resRef[numRows] = { 14.1, 12.5, 12.4, 8.3};
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, MultZero) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	TypeParam mult[numRows] = {};
	TypeParam add[numRows] = { 5,6,7,8 };
	const TypeParam resRef[numRows] = { 5,6,7,8 };
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultAdd, Basic) {
	const int numRows = CSMatrixRMultOp<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,0,3,4 };
	TypeParam add[numRows] = { 5,6,7,8,10 };
	const TypeParam resRef[numRows] = { 19.1, 12.7, 16., 15.5, 10 };
	CSMatrixRMultOp<TypeParam>::m.rMultAdd(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}


template<typename T>
using CSMatrixRMultSub = CSMatrixRMultOp<T>;

TYPED_TEST_SUITE(CSMatrixRMultSub, MyTypes);

TYPED_TEST(CSMatrixRMultSub, EmptyMatrix) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,2,3,4 };
	TypeParam add[numRows] = { 5,6,7,8 };
	const TypeParam resRef[numRows] = { 5,6,7,8 };
	SMM::TripletMatrix<TypeParam> emptyTriplet(4, 4, 10);
	SMM::CSRMatrix emptyMatrix(emptyTriplet);
	emptyMatrix.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubMultZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = {};
	TypeParam add[numRows] = {};
	const TypeParam resRef[numRows] = {};
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, SubZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,2,3,4 };
	TypeParam add[numRows] = {};
	const TypeParam resRef[numRows] = { -14.1, -12.5, -12.4, -8.3 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, MultZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = {};
	TypeParam add[numRows] = { 5,6,7,8 };
	const TypeParam resRef[numRows] = { 5,6,7,8 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, Basic) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = { 1,0,3,4 };
	TypeParam add[numRows] = { 5,6,7,8,10 };
	const TypeParam resRef[numRows] = { -9.1, -0.7, -2., 0.5, 10 };
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

// ==========================================================================
// ===================== COMPRESSED SPARSE ROW MATRIX  ======================
// ==========================================================================

TYPED_TEST(CSRMatrixTest, CSRMatrixEmptyConstForwardIterator) {
	SMM::TripletMatrix<TypeParam> triplet(10, 10);
	SMM::CSRMatrix csr(triplet);
	EXPECT_TRUE(csr.begin() == csr.end());
}

TYPED_TEST(CSRMatrixTest, CSRMatrixConstForwardIterator) {
	const int numRows = 6;
	const int numCols = 6;
	TypeParam denseRef[numRows][numCols] = {};
	denseRef[1][1] = 1.1;
	denseRef[1][2] = 2.2;
	denseRef[1][3] = 3.3;
	denseRef[4][4] = 4.4;

	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
	triplet.addEntry(4, 4, 4.4);
	triplet.addEntry(1, 1, 1.1);
	triplet.addEntry(1, 2, 2.2);
	triplet.addEntry(1, 3, 3.3);

	SMM::CSRMatrix<TypeParam> csr(triplet);
	typename SMM::CSRMatrix<TypeParam>::ConstIterator it = csr.begin();

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

template<typename T>
class CSRArithmetic : public testing::Test { };
TYPED_TEST_SUITE(CSRArithmetic, MyTypes);

TYPED_TEST(CSRArithmetic, MultiplyByScalar) {
	const int numRows = 6;
	const int numCols = 10;

	TypeParam denseRef[numRows][numCols] = {};
	denseRef[2][8] = 28.41637;
	denseRef[2][2] = 31.52779;
	denseRef[1][7] = -237.59453;
	denseRef[5][3] = 273.3937;
	denseRef[0][3] = -471.11824;

	SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
			}
		}
	}

	SMM::CSRMatrix<TypeParam> csr;
	csr.init(triplet);

	TypeParam scalar = -478.53439;

	csr *= scalar;

	for(const auto& el : csr) {
		EXPECT_EQ(el.getValue(), scalar * denseRef[el.getRow()][el.getCol()]);
	}
}


TYPED_TEST(CSRArithmetic, InplaceAdd) {
	const int numRows = 6;
	const int numCols = 10;

	TypeParam denseRef1[numRows][numCols] = {};
	denseRef1[2][8] = 28.41637;
	denseRef1[2][2] = 31.52779;
	denseRef1[1][7] = -237.59453;
	denseRef1[5][3] = 273.3937;
	denseRef1[0][3] = -471.11824;

	TypeParam denseRef2[numRows][numCols] = {};
	denseRef1[2][8] = 558.57004;
	denseRef1[2][2] = 53.47841;
	denseRef1[1][7] = 621.94377;
	denseRef1[5][3] = 237.2853;
	denseRef1[0][3] = -449.43152;

	SMM::TripletMatrix<TypeParam> triplet1(numRows, numCols);
	SMM::TripletMatrix<TypeParam> triplet2(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef1[i][j] != 0) {
				triplet1.addEntry(i, j, denseRef1[i][j]);
				triplet2.addEntry(i, j, denseRef2[i][j]);
			}
		}
	}

	SMM::CSRMatrix<TypeParam> csr1;
	csr1.init(triplet1);

	SMM::CSRMatrix<TypeParam> csr2;
	csr2.init(triplet2);

	csr1.inplaceAdd(csr2);

	for(const auto& el : csr1) {
		EXPECT_EQ(el.getValue(), denseRef1[el.getRow()][el.getCol()] + denseRef2[el.getRow()][el.getCol()]);
	}
}

TYPED_TEST(CSRArithmetic, InplaceSubtract) {
	const int numRows = 6;
	const int numCols = 10;

	TypeParam denseRef1[numRows][numCols] = {};
	denseRef1[2][8] = 28.41637;
	denseRef1[2][2] = 31.52779;
	denseRef1[1][7] = -237.59453;
	denseRef1[5][3] = 273.3937;
	denseRef1[0][3] = -471.11824;

	TypeParam denseRef2[numRows][numCols] = {};
	denseRef1[2][8] = 558.57004;
	denseRef1[2][2] = 53.47841;
	denseRef1[1][7] = 621.94377;
	denseRef1[5][3] = 237.2853;
	denseRef1[0][3] = -449.43152;

	SMM::TripletMatrix<TypeParam> triplet1(numRows, numCols);
	SMM::TripletMatrix<TypeParam> triplet2(numRows, numCols);
	for(int i = 0; i < numRows; ++i) {
		for(int j = 0; j < numCols; ++j) {
			if(denseRef1[i][j] != 0) {
				triplet1.addEntry(i, j, denseRef1[i][j]);
				triplet2.addEntry(i, j, denseRef2[i][j]);
			}
		}
	}

	SMM::CSRMatrix<TypeParam> csr1;
	csr1.init(triplet1);

	SMM::CSRMatrix<TypeParam> csr2;
	csr2.init(triplet2);

	csr1.inplaceSubtract(csr2);

	for(const auto& el : csr1) {
		EXPECT_EQ(el.getValue(), denseRef1[el.getRow()][el.getCol()] - denseRef2[el.getRow()][el.getCol()]);
	}
}

// =========================================================================
// ============================== FILE LOAD ================================
// =========================================================================

template<typename T>
class LoadFile : public testing::Test { };
TYPED_TEST_SUITE(LoadFile, MyTypes);

TYPED_TEST(LoadFile, LoadMatrixMarket) {
	const std::string path = ASSET_PATH + std::string("mesh1e1_structural_48_48_177.mtx");
	SMM::TripletMatrix<TypeParam> triplet;
	const SMM::MatrixLoadStatus status = SMM::loadMatrix(path.c_str(), triplet);
	ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
	EXPECT_EQ(triplet.getDenseRowCount(), 48);
	EXPECT_EQ(triplet.getDenseColCount(), 48);
	EXPECT_EQ(triplet.getNonZeroCount(), 306);
}

// =========================================================================
// ============================== FILE SAVE ================================
// =========================================================================

template<typename T>
class SaveDense : public testing::Test { };
TYPED_TEST_SUITE(SaveDense, MyTypes);

TYPED_TEST(SaveDense, CSRSMMDT) {
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
	SMM::TripletMatrix<TypeParam> triplet(10, 10);
	triplet.addEntry(0, 0, 234.5324);
	triplet.addEntry(3, 2, 2.4);
	triplet.addEntry(5, 2, 1);
	triplet.addEntry(5, 3, 2);
	triplet.addEntry(5, 4, 3);
	triplet.addEntry(5, 6, 4);
	triplet.addEntry(6, 1, 3.4);
	for (int i = 0; i < 10; ++i) triplet.addEntry(9, i, 1);
	SMM::CSRMatrix<TypeParam> csr(triplet);
	FileRemoveRAII file(ASSET_PATH + std::string("__test.smmdt"));
	SMM::saveDenseText(file.getName(), csr);

	SMM::TripletMatrix<TypeParam> tripletRead;
	const SMM::MatrixLoadStatus status = SMM::loadMatrix(file.getName(), tripletRead);
	ASSERT_EQ(status, SMM::MatrixLoadStatus::SUCCESS);
	EXPECT_TRUE(NearMatrix(tripletRead, csr, TypeParam(1e-6)));
}

int main(int argc, char **argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}