#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest/doctest.h"
#include "sparse_matrix_math.h"

TEST_SUITE("TripletMatrix") {
	TEST_CASE("TripletMatrix constructor with given rows and columns") {
		const int numRows = 5;
		const int numCols = 10;
		SMM::TripletMatrix<float> m(numRows, numCols);
		CHECK_EQ(m.getDenseRowCount(), numRows);
		CHECK_EQ(m.getDenseColCount(), numCols);
		CHECK_EQ(m.getNonZeroCount(), 0);
	}

	TEST_CASE("TripletMatrix constructor with given rows, columns and number NNZ elements to preallocate") {
		const int numRows = 5;
		const int numCols = 10;
		const int numElements = 5;
		SMM::TripletMatrix<float> m(numRows, numCols, numElements);
		CHECK_EQ(m.getDenseRowCount(), numRows);
		CHECK_EQ(m.getDenseColCount(), numCols);
		CHECK_EQ(m.getNonZeroCount(), 0);
	}
	TEST_CASE("TripletMatrix adding elements") {
		const int numRows = 10;
		const int numCols = 12;
		SMM::TripletMatrix<float> m(numRows, numCols);
		CHECK_EQ(m.getNonZeroCount(), 0);

		m.addEntry(1, 1, 1.0);
		CHECK_EQ(m.getNonZeroCount(), 1);

		m.addEntry(2, 2, 1.0);
		CHECK_EQ(m.getNonZeroCount(), 2);

		m.addEntry(1, 1, 5.0);
		CHECK_EQ(m.getNonZeroCount(), 2);

		m.addEntry(3, 5, 10.0);
		CHECK_EQ(m.getNonZeroCount(), 3);
	}


	TEST_CASE("TripletMatrix add elements when constructor with preallocation was added") {
		const int numRows = 10;
		const int numCols = 12;
		const int numElements = 2;
		SMM::TripletMatrix<float> m(numRows, numCols, numElements);
		CHECK_EQ(m.getNonZeroCount(), 0);

		m.addEntry(1, 1, 1.0);
		CHECK_EQ(m.getNonZeroCount(), 1);

		m.addEntry(2, 2, 1.0);
		CHECK_EQ(m.getNonZeroCount(), 2);

		m.addEntry(1, 1, 5.0);
		CHECK_EQ(m.getNonZeroCount(), 2);

		m.addEntry(3, 5, 10.0);
		CHECK_EQ(m.getNonZeroCount(), 3);
	}

	TEST_CASE_TEMPLATE("Convert TripletMatrix to linear dense matrix", TypeParam, float, double) {
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
			CHECK_EQ(dense[i], doctest::Approx(denseRef[i]).epsilon(10e-6));
		}
	}

	TEST_CASE_TEMPLATE("Access TripletMatrix elements by row and col", TypeParam, float, double) {
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

		for (int i = 0; i < numRows; ++i) {
			for (int j = 0; j < numCols; ++j) {
				CHECK_EQ(m.getValue(i, j), doctest::Approx(denseRef[i * numCols + j]).epsilon(1e-6));
			}
		}

		SUBCASE("Updating non existing element returns false and does nothing") {
			CHECK_FALSE(m.updateEntry(0, 1, 2));
			CHECK_EQ(m.getValue(0, 1), TypeParam(0));
		}

		// Update existing element
		SUBCASE("Updating existing element returns true and updates the element") {
			CHECK(m.updateEntry(1, 3, TypeParam(200)));
			CHECK_EQ(m.getValue(1, 3), TypeParam(200));

		}
	}
}