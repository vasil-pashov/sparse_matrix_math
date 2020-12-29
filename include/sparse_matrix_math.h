#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <utility>

namespace SparseMatrix {
	/// @brief Class to hold sparse matrix into triplet (coordinate) format.
	/// Triplet format represents the matrix entries as list of triplets (row, col, value)
	/// It is allowed repetition of elements, i.e. row and col can be the same for two
	/// separate entries, later on when the matrix is converted to compresses sparse format elements with
	/// repeating indexes will be summed. This class is supposed to be used as intermediate class to add
	/// entries dynamically. After all data has been added call toCSR or toCSC to get sparse
	/// matrix in compressed sparse row or compressed sparse column format.
	class TripletMatrix {
	private:
		struct Triplet {
			friend class TripletMatrix;
			Triplet(int row, int col, float value) noexcept :
				row(row),
				col(col),
				value(value)
			{ }
			const int getRow() const noexcept {
				return row;
			}
			const int getCol() const noexcept {
				return col;
			}
			const int getValue() const noexcept {
				return value;
			}
		private:
			int row, col;
			float value;
		};
	public:
		using ConstIterator = std::vector<Triplet>::const_iterator;
		/// @brief Create empty triplet matrix
		TripletMatrix() = default;
		/// @brief Create triplet matrix with predefined number of triplets in it.
		/// @param numTriplets Number of triples which will be allocated.
		TripletMatrix(size_t numTriplets);
		~TripletMatrix() = default;
		TripletMatrix(TripletMatrix&&) = default;
		TripletMatrix& operator=(TripletMatrix&&) = default;
		TripletMatrix(const TripletMatrix&) = delete;
		TripletMatrix& operator=(const TripletMatrix&) = delete;
		/// @brief Add triplet entry to the matrix
		/// @param row Row of the element
		/// @param col Column of the element
		/// @param value The value of the element at (row, col)
		void addEntry(int row, int col, float value);
		/// @brief Get constant iterator to the first element of the triplet list
		/// @return Constant iterator to the first element of the triplet list
		ConstIterator begin() const noexcept;
		/// @brief Get constant iterator to the end of the triplet list
		/// @return Constant iterator to the end of the triplet list
		ConstIterator end() const noexcept;
		/// @brief Get the number of triplets
		/// @return Number of triplets into the array
		const size_t size() const noexcept;
	private:
		/// @brief List of triplets. Array of structures is chosen since converting to
		/// compressed sparse format requires iteration over all triplets and this format
		/// is more cache friendly.
		std::vector<Triplet> data;
		/// @brief Keep track of repeating indexes
		/// Key is unique representation of the matrix index (first 32 bits are the col second are the row)
		/// The value is index into the data array of triplets where the element is
		std::unordered_map<uint64_t, int> entryIndex;
	};

	TripletMatrix::TripletMatrix(size_t numTriplets) {
		data.reserve(numTriplets);
	}

	void TripletMatrix::addEntry(int row, int col, float value) {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		const uint64_t key = static_cast<uint64_t>(row) << sizeof(int) | static_cast<uint64_t>(col);
		auto it = entryIndex.find(key);
		if (it == entryIndex.end()) {
			const int tripletIndex = data.size();
			data.emplace_back(row, col, value);
			entryIndex[key] = tripletIndex;
		} else {
			data[it->second].value += value;
		}
	}

	inline TripletMatrix::ConstIterator TripletMatrix::begin() const noexcept {
		return data.begin();
	}

	inline TripletMatrix::ConstIterator TripletMatrix::end() const noexcept {
		return data.end();
	}

	inline const size_t TripletMatrix::size() const noexcept {
		return data.size();
	}
	/// @brief Const forward iterator for matrix in compressed sparse row format
	class CSRConstIterator {
	private:
		class CSRElement {
		friend class CSRConstIterator;
		public:
			const int getRow() const noexcept;
			const int getCol() const noexcept;
			const float getValue() const noexcept;
			friend void swap(CSRElement& a, CSRElement& b) noexcept;
		private:
			CSRElement(
				const float* values,
				const int* columnIndex,
				const int* rowPointer,
				const int currentRow,
				const int currentColumIndex
			) noexcept;
			const bool operator==(const CSRElement& other) const noexcept;

			CSRElement(const CSRElement&) = default;
			CSRElement& operator=(const CSRElement&) = default;

			const float* values;
			const int* columnIndex;
			const int* rowPointer;
			int currentRow;
			int currentColumnIndex;
		};
		friend void swap(CSRElement& a, CSRElement& b) noexcept;
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = CSRElement;
		using pointer = const CSRElement*;
		using reference = const CSRElement&;

		CSRConstIterator() = default;
		CSRConstIterator(
			const float* values,
			const int* columnPointer,
			const int* rowPointer,
			const int currentRow,
			const int currentColumnIndex
		) noexcept;
		CSRConstIterator(const CSRConstIterator&) = default;
		CSRConstIterator& operator=(const CSRConstIterator&) = default;
		reference operator*() const;
		pointer operator->() const;
		const bool operator==(const CSRConstIterator& other) const noexcept;
		const bool operator!=(const CSRConstIterator& other) const noexcept;
		CSRConstIterator& operator++() noexcept;
		CSRConstIterator operator++(int) noexcept;
		friend void swap(CSRConstIterator& a, CSRConstIterator& b) noexcept;
	private:
		CSRElement currentElement;
	};

	CSRConstIterator::CSRElement::CSRElement(
		const float* values,
		const int* columnIndex,
		const int* rowPointer,
		const int currentRow,
		const int currentColumIndex
	) noexcept :
		values(values),
		columnIndex(columnIndex),
		rowPointer(rowPointer),
		currentRow(currentRow),
		currentColumnIndex(currentColumnIndex) {
	}

	const int CSRConstIterator::CSRElement::getRow() const noexcept {
		return currentRow;
	}

	const int CSRConstIterator::CSRElement::getCol() const noexcept {
		return columnIndex[currentColumnIndex];
	}

	const float CSRConstIterator::CSRElement::getValue() const noexcept {
		return values[currentColumnIndex];
	}

	const bool CSRConstIterator::CSRElement::operator==(const CSRElement& other) const noexcept {
		return values == other.values &&
			columnIndex == other.columnIndex &&
			rowPointer == other.rowPointer &&
			currentRow == other.currentRow &&
			currentColumnIndex == other.currentColumnIndex;
	}

	void swap(CSRConstIterator::CSRElement& a, CSRConstIterator::CSRElement& b) noexcept {
		using std::swap;
		swap(a.values, b.values);
		swap(a.columnIndex, b.columnIndex);
		swap(a.rowPointer, b.rowPointer);
		swap(a.currentRow, b.currentRow);
		swap(a.currentColumnIndex, b.currentColumnIndex);
	}

	CSRConstIterator::CSRConstIterator(
		const float* values,
		const int* columnIndex,
		const int* rowPointer,
		const int currentRow,
		const int currentColumIndex
	) noexcept :
		currentElement(values, columnIndex, rowPointer, currentRow, currentColumIndex)
	{ }

	inline CSRConstIterator::reference CSRConstIterator::operator*() const {
		return currentElement;
	}

	inline CSRConstIterator::pointer CSRConstIterator::operator->() const {
		return &currentElement;
	}

	inline const bool CSRConstIterator::operator==(const CSRConstIterator& other) const noexcept {
		return currentElement == other.currentElement;
	}

	inline const bool CSRConstIterator::operator!=(const CSRConstIterator& other) const noexcept {
		return !(*this == other);
	}

	CSRConstIterator& CSRConstIterator::operator++() noexcept {
		currentElement.currentColumnIndex++;
		if (currentElement.currentColumnIndex >= currentElement.rowPointer[currentElement.currentRow + 1]) {
			currentElement.currentRow++;
		}
		return *this;
	}

	CSRConstIterator CSRConstIterator::operator++(int) noexcept {
		CSRConstIterator initialState = *this;
		++(*this);
		return initialState;
	}

	inline void swap(CSRConstIterator& a, CSRConstIterator& b) noexcept {
		swap(a.currentElement, b.currentElement);
	}

	/// @brief Matrix in compressed sparse row format
	/// Compressed sparse row format is represented with 3 arrays. One for the values,
	/// one to keep track nonzero columns for each row, one to keep track where each row starts
	/// The columns in each row are not ordered in any particular way (i.e. in ascending order)
	class CSRMatrix {
	public:
		using LinearizedDenseFormat = std::unique_ptr<float[], decltype(&std::free)>;
		/// @brief Initialize matrix in compressed sparse row format from a given matrix into triplet format
		/// In case that the constructor could not allocate the needed amount of memory,
		/// denseRowCount and denseColCount are set to -1 and all memory that was allocated will be freed.
		/// @param rows Number of rows in the matrix
		/// @param cols Number of columns in the matrix
		/// @param tripletMatrix Matrix in triplet format from which CSR matrix will be created
		CSRMatrix(int rows, int cols, const TripletMatrix& tripletMatrix) noexcept;
		CSRMatrix(CSRMatrix&&) noexcept;
		CSRMatrix& operator=(CSRMatrix&&) noexcept;
		CSRMatrix(const CSRMatrix&) = delete;
		CSRMatrix& operator=(const CSRMatrix&) = delete;
		/// @brief Get the number of trivial nonzero entries in the matrix
		/// Trivial nonzero entries do not include zero elements which came from numerical cancellation
		/// @return The number of trivial nonzero entries in the matrix
		const int getNonZeroCount() const noexcept;
		/// @brief Get the total number of rows of the matrix
		/// @return The row count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseRowCount() const noexcept;
		/// @brief Get the total number of cols of the matrix
		/// @return The column count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseColCount() const noexcept;
		/// @brief Allocate and fill dense version of the matrix
		/// @return Pointer to the allocated matrix.
		LinearizedDenseFormat toLinearizedDenseMatrix() const;
		/// @brief Get forward iterator to the starting from the beginning of the matrix
		/// The rows are guaranteed to be iterated in nondecreasing order
		/// There is no particular order in which the columns will be iterated
		/// @return Forward iterator to the beginning of the matrix
		CSRConstIterator begin() const;
		/// @brief Iterator to one element past the last element in the matrix
		/// Dereferencing this iterator results in undefined behavior.
		CSRConstIterator end() const;
	private:
		/// @brief Array which will hold all nonzero entries of the matrix
		/// This is of length  number of nonzero entries
		std::unique_ptr<float[]> values;
		/// @brief Elements of this array represent indexes of columns where nonzero values are
		/// It begins with the columns of the nonzero entries of the first row (if any), then the second
		/// and so on. This is of length number of nonzero entries
		std::unique_ptr<int[]>(columnIndex);
		/// @brief Elements of this array the indexes where the i-th row begins in the values and columnIndex arrays
		/// This is of length number of rows
		std::unique_ptr<int[]> rowPointer;
		int denseRowCount; ///< Number of rows in the matrix
		int denseColCount; ///< Number of columns in the matrix
	};

	CSRMatrix::CSRMatrix(const int rows, const int cols, const TripletMatrix& triplet) noexcept :
		values(new float[triplet.size()]),
		columnIndex(new int[triplet.size()]),
		rowPointer(new int[rows + 1]),
		denseRowCount(rows),
		denseColCount(cols)
	{
		// Calloc was proven to be faster than new [rows]()
		std::unique_ptr<int[], decltype(&std::free)> rowCount(static_cast<int*>(calloc(rows, sizeof(int))), &std::free);
		if (values == nullptr || columnIndex == nullptr || rowPointer == nullptr || rowCount == nullptr) {
			assert(false);
			denseRowCount = -1;
			denseColCount = -1;
			values.reset();
			columnIndex.reset();
			rowPointer.reset();
			rowCount.reset();
			return;
		}

		for (const auto& el : triplet) {
			rowCount[el.getRow()]++;
		}

		rowPointer[0] = 0;
		for (int i = 1; i < rows; ++i) {
			rowPointer[i] += rowCount[i - 1];
		}
		rowPointer[rows] = triplet.size();

		for (const auto& el : triplet) {
			const int row = el.getRow();
			const int index = rowPointer[row + 1] - rowCount[row];
			values[index] = el.getValue();
			columnIndex[index] = el.getCol();
			rowCount[row]--;
		}
	}

	CSRMatrix::CSRMatrix(CSRMatrix&& other) noexcept :
		values(std::move(other.values)),
		columnIndex(std::move(other.columnIndex)),
		rowPointer(std::move(other.rowPointer)),
		denseRowCount(other.denseRowCount),
		denseColCount(other.denseColCount)
	{
		other.denseRowCount = 0;
		other.denseColCount = 0;
	}

	CSRMatrix& CSRMatrix::operator=(CSRMatrix&& other) noexcept {
		values = std::move(other.values);
		columnIndex = std::move(other.columnIndex);
		rowPointer = std::move(other.rowPointer);
		denseRowCount = other.denseRowCount;
		denseColCount = other.denseColCount;
		other.denseRowCount = 0;
		other.denseColCount = 0;
		return *this;
	}

	inline const int CSRMatrix::getNonZeroCount() const noexcept {
		return rowPointer[denseRowCount];
	}

	inline const int CSRMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline const int CSRMatrix::getDenseColCount() const noexcept {
		return denseColCount;
	}

	inline CSRMatrix::LinearizedDenseFormat CSRMatrix::toLinearizedDenseMatrix() const {
		int64_t size = int64_t(denseRowCount) * denseColCount;
		LinearizedDenseFormat dense(static_cast<float*>(calloc(size, sizeof(float))), &std::free);
		if (dense == nullptr) {
			return dense;
		}
		assert(dense[0] == 0.0f);
		for (int i = 0; i < denseRowCount; ++i) {
			for (int j = rowPointer[i]; j < rowPointer[i + 1]; ++j) {
				const int column = columnIndex[j];
				const int index = i * denseColCount + column;
				assert(index < size);
				dense[index] = values[j];
			}
		}
		return dense;
	}

	inline CSRConstIterator CSRMatrix::begin() const {
		return CSRConstIterator(values.get(), columnIndex.get(), rowPointer.get(), 0, 0);
	}

	inline CSRConstIterator CSRMatrix::end() const {
		const int nonZeroCount = getNonZeroCount();
		return CSRConstIterator(
			values.get(),
			columnIndex.get(),
			rowPointer.get(),
			denseRowCount,
			nonZeroCount
		);
	}
}