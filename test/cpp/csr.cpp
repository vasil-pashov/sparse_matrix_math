#include "doctest/doctest.h"
#include "sparse_matrix_math.h"

TEST_SUITE("CSR Matrix Constructor") {
	TEST_CASE_TEMPLATE("CSRMatrix empty constructor", TypeParam, float, double) {
		SMM::CSRMatrix<TypeParam> m;
		CHECK_EQ(m.getDenseRowCount(), 0);
		CHECK_EQ(m.getDenseColCount(), 0);
		CHECK_EQ(m.getNonZeroCount(), 0);
	}

	TEST_CASE_TEMPLATE("Construct CSR Matrix from empty triplet", TypeParam, float, double) {
		const int numRows = 4;
		const int numCols = 5;
		SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
		SUBCASE("Call CSRMatrix constructor") {
			SMM::CSRMatrix m(triplet);
			CHECK_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
			CHECK_EQ(m.getDenseColCount(), triplet.getDenseColCount());
			CHECK_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
		}

		SUBCASE("Call init function") {
			SMM::CSRMatrix<TypeParam> m;
			CHECK_EQ(m.init(triplet), 0);
			CHECK_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
			CHECK_EQ(m.getDenseColCount(), triplet.getDenseColCount());
			CHECK_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
		}
	}

	TEST_CASE_TEMPLATE("Construct CSR Matrix from non empty triplet", TypeParam, float, double) {
		const int numRows = 10;
		const int numCols = 12;
		const int numElements = 2;
		SMM::TripletMatrix<TypeParam> triplet(numRows, numCols, numElements);
		triplet.addEntry(1, 1, 1.0);
		triplet.addEntry(2, 2, 1.0);
		triplet.addEntry(1, 1, 5.0);
		triplet.addEntry(3, 5, 10.0);

		SUBCASE("Call CSRMatrix constructor") {
			SMM::CSRMatrix csr(triplet);
			CHECK_EQ(csr.getDenseRowCount(), triplet.getDenseRowCount());
			CHECK_EQ(csr.getDenseColCount(), triplet.getDenseColCount());
			CHECK_EQ(csr.getNonZeroCount(), triplet.getNonZeroCount());
		}

		SUBCASE("Call init function") {
			SMM::CSRMatrix<TypeParam> m;
			CHECK_EQ(m.init(triplet), 0);
			CHECK_EQ(m.getDenseRowCount(), triplet.getDenseRowCount());
			CHECK_EQ(m.getDenseColCount(), triplet.getDenseColCount());
			CHECK_EQ(m.getNonZeroCount(), triplet.getNonZeroCount());
		}
	}
}

TEST_SUITE("CSRMatrix direct element access") {
	TEST_CASE_TEMPLATE("CSRMatrix direct element access", TypeParam, float, double) {
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

		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				CHECK_EQ(m.getValue(i, j), doctest::Approx(denseRef[i * numCols + j]).epsilon(1e-6));
			}
		}

		SUBCASE("Update non existing element") {
			CHECK_FALSE(m.updateEntry(0, 1, 2));
			CHECK_EQ(m.updateEntry(0, 1, 2), TypeParam(0));
		}

		SUBCASE("Update existing element") {
			CHECK(m.updateEntry(1, 3, TypeParam(200)));
			CHECK_EQ(m.getValue(1, 3), TypeParam(200));
		}
	}
}

TEST_SUITE("CSRMatrix iterators") {
	TEST_CASE_TEMPLATE("Begin and end iterators on empty matrix", TypeParam, float, double) {
		{
			SMM::CSRMatrix<TypeParam> csr;
			SUBCASE("Non-const iterator") {
				CHECK(csr.begin() == csr.end());
			}

			SUBCASE("Const iterator") {
				CHECK(csr.cbegin() == csr.cend());
			}
		}
	}
	TEST_CASE_TEMPLATE("Begin and end iterators on CSR matrix created from empty TripletMatrix", TypeParam, float, double) {
		{
			SMM::TripletMatrix<TypeParam> triplet(10, 10);
			SMM::CSRMatrix<TypeParam> csr(triplet);
			SUBCASE("Non-const iterator") {
				CHECK(csr.begin() == csr.end());
			}
			SUBCASE("Const iterator") {
				CHECK(csr.cbegin() == csr.cend());
			}
		}
	}

	TEST_CASE_TEMPLATE("CSRMatrix const iterator can read the whole matrix", TypeParam, float, double) {
		const int numRows = 6;
		const int numCols = 6;
		TypeParam denseRef[numRows][numCols] = {};
		denseRef[1][1] = 1.1;
		denseRef[1][2] = 2.2;
		denseRef[1][5] = 3.3;
		denseRef[4][4] = 4.4;

		SMM::TripletMatrix<TypeParam> triplet(numRows, numCols);
		triplet.addEntry(1, 1, 1.1);
		triplet.addEntry(1, 2, 2.2);
		triplet.addEntry(1, 5, 3.3);
		triplet.addEntry(4, 4, 4.4);

		SMM::CSRMatrix<TypeParam> csr(triplet);
		typename SMM::CSRMatrix<TypeParam>::ConstIterator it = csr.begin();

		CHECK_EQ(it->getRow(), 1);
		CHECK_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

		++it;
		CHECK_EQ(it->getRow(), 1);
		CHECK_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

		++it;
		CHECK_EQ(it->getRow(), 1);
		CHECK_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

		++it;
		CHECK_EQ(it->getRow(), 4);
		CHECK_EQ(it->getValue(), denseRef[it->getRow()][it->getCol()]);

		++it;
		CHECK(it == csr.end());
	}

	TEST_CASE_TEMPLATE("CSRMatrix row iterator", TypeParam, float, double) {
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
		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				if (denseRef[i][j] != 0) {
					triplet.addEntry(i, j, denseRef[i][j]);
					numElementsInRow[i]++;
				}
			}
		}

		SMM::CSRMatrix<TypeParam> m;
		m.init(triplet);

		SUBCASE("Can read each row") {
			for (int i = 0; i < numRows; ++i) {
				int numElements = 0;
				typename SMM::CSRMatrix<TypeParam>::ConstRowIterator current = m.rowBegin(i);
				typename SMM::CSRMatrix<TypeParam>::ConstRowIterator rowEnd = m.rowEnd(i);
				while (current != rowEnd) {
					CHECK_EQ(current->getValue(), denseRef[current->getRow()][current->getCol()]);
					numElements++;
					++current;
				}
				CHECK_EQ(numElements, numElementsInRow[i]);
			}
		}

		SUBCASE("Can update values") {
			typename SMM::CSRMatrix<TypeParam>::RowIterator row = m.rowBegin(1);
			row->setValue(8);
			CHECK_EQ(row->getValue(), 8);
			CHECK_EQ(m.getValue(row->getRow(), row->getCol()), row->getValue());
		}
	}
}

TEST_SUITE("CSRMatrix conversion") {
	TEST_CASE_TEMPLATE("Convert CSRMatrix to linearized dense matrix", TypeParam, float, double) {
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
		CHECK_EQ(triplet.getNonZeroCount(), csr.getNonZeroCount());

		TypeParam dense[denseSize] = {};
		SMM::toLinearDenseRowMajor(csr, dense);
		for (int i = 0; i < denseSize; ++i) {
			CHECK_EQ(dense[i], denseRef[i]);
		}
	}
}

TEST_SUITE("CSRMatrix - Vector operations") {
	TEST_CASE_TEMPLATE("CSRMatrix saxpy", TypeParam, float, double) {
		const int numRows = 5;
		const int numCols = 4;
		const int numElements = 10;
		SMM::TripletMatrix<TypeParam> triplet(numRows, numCols, numElements);
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
		SMM::CSRMatrix<TypeParam> m;
		m.init(triplet);


		SUBCASE("A * x + b, A==0, b !=0, x != 0") {
			TypeParam mult[numRows] = {1,2,3,4,5};
			TypeParam add[numRows] = {5,6,7,8,9};
			TypeParam resRef[numRows] = {5,6,7,8,9};
			TypeParam res[numRows] = {};
			SMM::TripletMatrix<TypeParam> emptyTriplet(numRows, numCols, numElements);
			SMM::CSRMatrix emptyMatrix(emptyTriplet);
			emptyMatrix.rMultAdd(add, mult, res);
			for (int i = 0; i < numRows; ++i) {
				CHECK_EQ(res[i], resRef[i]);
			}
		}

		SUBCASE("A * x + b, A != 0, b == 0, x == 0") {
			TypeParam mult[numRows] = {};
			TypeParam add[numRows] = {};
			const TypeParam resRef[numRows] = {};
			SUBCASE("Result is inplace") {
				m.rMultAdd(add, mult, add);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], add[i]);
				}
			}

			SUBCASE("Result is in separate vector") {
				TypeParam res[numRows] = {};
				m.rMultAdd(add, mult, res);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], res[i]);
				}
			}
		}

		SUBCASE("A * x + b, A != 0, x != 0, b == 0") {
			TypeParam mult[numRows] = {1,2,3,4,5};
			TypeParam add[numRows] = {};
			const TypeParam resRef[numRows] = {14.1, 12.5, 12.4, 8.3,0};
			SUBCASE("Result is inplace") {
				m.rMultAdd(add, mult, add);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(add[i]).epsilon(1e-6));
				}
			}

			SUBCASE("Result is in separate vector") {
				TypeParam res[numRows] = {};
				m.rMultAdd(add, mult, res);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(res[i]).epsilon(1e-6));
				}
			}
		}

		SUBCASE("A * x + b, A != 0, x == 0, b != 0") {
			TypeParam mult[numRows] = {};
			TypeParam add[numRows] = {5,6,7,8,9};
			const TypeParam resRef[numRows] = {5,6,7,8,9};
			SUBCASE("Result is inplace") {
				m.rMultAdd(add, mult, add);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(add[i]).epsilon(1e-6));
				}
			}

			SUBCASE("Result is in separate vector") {
				TypeParam res[numRows] = {};
				m.rMultAdd(add, mult, res);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(res[i]).epsilon(1e-6));
				}
			}
		}

		SUBCASE("A * x + b, A !=0, x != 0, b != 0") {
			TypeParam mult[numRows] = {1,0,3,4};
			TypeParam add[numRows] = {5,6,7,8,10};
			const TypeParam resRef[numRows] = {19.1, 12.7, 16., 15.5, 10};
			SUBCASE("Result is inplace") {
				m.rMultAdd(add, mult, add);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(add[i]).epsilon(1e-6));
				}
			}

			SUBCASE("Result is in separate vector") {
				TypeParam res[numRows] = {};
				m.rMultAdd(add, mult, res);
				for (int i = 0; i < numRows; ++i) {
					CHECK_EQ(resRef[i], doctest::Approx(res[i]).epsilon(1e-6));
				}
			}
		}
	}
}

#if 0

TYPED_TEST_SUITE(CSMatrixRMultSub, MyTypes);

TYPED_TEST(CSMatrixRMultSub, EmptyMatrix) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = {1,2,3,4};
	TypeParam add[numRows] = {5,6,7,8};
	const TypeParam resRef[numRows] = {5,6,7,8};
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
	TypeParam mult[numRows] = {1,2,3,4};
	TypeParam add[numRows] = {};
	const TypeParam resRef[numRows] = {-14.1, -12.5, -12.4, -8.3};
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, MultZero) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = {};
	TypeParam add[numRows] = {5,6,7,8};
	const TypeParam resRef[numRows] = {5,6,7,8};
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}

TYPED_TEST(CSMatrixRMultSub, Basic) {
	const int numRows = CSMatrixRMultSub<TypeParam>::numRows;
	TypeParam mult[numRows] = {1,0,3,4};
	TypeParam add[numRows] = {5,6,7,8,10};
	const TypeParam resRef[numRows] = {-9.1, -0.7, -2., 0.5, 10};
	CSMatrixRMultSub<TypeParam>::m.rMultSub(add, mult, add);
	for (int i = 0; i < numRows; ++i) {
		EXPECT_NEAR(resRef[i], add[i], 1e-6);
	}
}



// =========================================================================
// ======================= Arithmetic operations ===========================
// =========================================================================

template<typename T>
class CSRArithmetic : public testing::Test {
};
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
	for (int i = 0; i < numRows; ++i) {
		for (int j = 0; j < numCols; ++j) {
			if (denseRef[i][j] != 0) {
				triplet.addEntry(i, j, denseRef[i][j]);
			}
		}
	}

	SMM::CSRMatrix<TypeParam> csr;
	csr.init(triplet);

	TypeParam scalar = -478.53439;

	csr *= scalar;

	for (const auto& el : csr) {
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
	for (int i = 0; i < numRows; ++i) {
		for (int j = 0; j < numCols; ++j) {
			if (denseRef1[i][j] != 0) {
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

	for (const auto& el : csr1) {
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
	for (int i = 0; i < numRows; ++i) {
		for (int j = 0; j < numCols; ++j) {
			if (denseRef1[i][j] != 0) {
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

	for (const auto& el : csr1) {
		EXPECT_EQ(el.getValue(), denseRef1[el.getRow()][el.getCol()] - denseRef2[el.getRow()][el.getCol()]);
	}
}

// =========================================================================
// ============================== FILE LOAD ================================
// =========================================================================

template<typename T>
class LoadFile : public testing::Test {
};
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
class SaveDense : public testing::Test {
};
TYPED_TEST_SUITE(SaveDense, MyTypes);

TYPED_TEST(SaveDense, CSRSMMDT) {
	struct FileRemoveRAII {
		FileRemoveRAII(const std::string& name) :
			name(name) {
		}
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

int main(int argc, char** argv) {
	testing::InitGoogleTest(&argc, argv);
	return RUN_ALL_TESTS();
}
#endif