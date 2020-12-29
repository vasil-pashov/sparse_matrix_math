#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <utility>
#include <cinttypes>

namespace SparseMatrix {
	/// @brief Class to hold sparse matrix into triplet (coordinate) format.
	/// Triplet format represents the matrix entries as list of triplets (row, col, value)
	/// It is allowed repetition of elements, i.e. row and col can be the same for two
	/// separate entries, when this happens elements are being summed. Repeating elements does not
	/// increase the count of non zero elements. This class is supposed to be used as intermediate class to add
	/// entries dynamically. After all data is gathered it should be converted to CSRMatrix which provides
	/// various arithmetic functions.
	class TripletMatrix {
	private:
		struct Triplet {
			friend class TripletMatrix;
			Triplet(int row, int col, float value) noexcept :
				row(row),
				col(col),
				value(value)
			{ }
			Triplet(const Triplet&) noexcept = default;
			Triplet(Triplet&&) noexcept = default;
			const int getRow() const noexcept {
				return row;
			}
			const int getCol() const noexcept {
				return col;
			}
			const float getValue() const noexcept {
				return value;
			}
		private:
			int row, col;
			float value;
		};
	public:
		using ConstIterator = std::vector<Triplet>::const_iterator;
		/// @brief Initialize triplet matrix with given number of rows and columns
		/// The number of rows and columns does not have any affect the space allocated by the matrix
		/// @param[in] rowCount Number of rows which the dense form of the matrix is supposed to have
		/// @param[in] colCount Number of columns which the dense form of the matrix is supposed to have
		TripletMatrix(int rowCount, int colCount) noexcept;
		/// @brief Initialize triplet matrix with given number of rows and columns and allocate space for the elements of the matrix
		/// Note that this constructor only allocates space but does not initialize the elements, nor it changes the number of non zero elements,
		/// thus the number of non zero elements will be 0 after the constructor is called.
		/// The number of rows and columns does not have any affect the space allocated by the matrix
		/// @param[in] rowCount Number of rows which the dense form of the matrix is supposed to have
		/// @param[in] colCount Number of columns which the dense form of the matrix is supposed to have
		/// @param[in] numTriplets How many elements to allocate space for.
		TripletMatrix(int rowCount, int colCount, int numTriplets) noexcept;
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
		const int getNonZeroCount() const noexcept;
		/// @brief Get the total number of rows of the matrix
		/// @return The row count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseRowCount() const noexcept;
		/// @brief Get the total number of cols of the matrix
		/// @return The column count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseColCount() const noexcept;
	private:
		/// @brief List of triplets. Array of structures is chosen since converting to
		/// compressed sparse format requires iteration over all triplets and this format
		/// is more cache friendly.
		std::vector<Triplet> data;
		/// @brief Keep track of repeating indexes
		/// Key is unique representation of the matrix index (first 32 bits are the col second are the row)
		/// The value is index into the data array of triplets where the element is
		std::unordered_map<uint64_t, int> entryIndex;
		int denseRowCount; ///< Number of rows in the matrix  
		int denseColCount; ///< Number of columns in the matrix
	};

	TripletMatrix::TripletMatrix(int denseRowCount, int denseColCount) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{ }

	TripletMatrix::TripletMatrix(int denseRowCount, int denseColCount, int numTriplets) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{
		data.reserve(numTriplets);
	}

	void TripletMatrix::addEntry(int row, int col, float value) {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && row < denseColCount);
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

	inline const int TripletMatrix::getNonZeroCount() const noexcept {
		return data.size();
	}

	inline const int TripletMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline const int TripletMatrix::getDenseColCount() const noexcept {
		return denseColCount;
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
		const int currentColumnIndex
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
		assert(currentElement.currentColumnIndex <= currentElement.rowPointer[currentElement.currentRow + 1]);
		if (currentElement.currentColumnIndex == currentElement.rowPointer[currentElement.currentRow + 1]) {
			do {
				currentElement.currentRow++;
			} while (currentElement.rowPointer[currentElement.currentRow + 1] == currentElement.currentColumnIndex);
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
		using ConstIterator = CSRConstIterator;
		/// @brief Initialize matrix in compressed sparse row format from a given matrix into triplet format
		/// In case that the constructor could not allocate the needed amount of memory,
		/// denseRowCount and denseColCount are set to -1 and all memory that was allocated will be freed.
		/// @param[in] tripletMatrix Matrix in triplet format from which CSR matrix will be created
		explicit CSRMatrix(const TripletMatrix& tripletMatrix) noexcept;
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
		int firstActiveRow; ///< Index of the first row which contains nonzero elements
	};

	CSRMatrix::CSRMatrix(const TripletMatrix& triplet) noexcept :
		values(new float[triplet.getNonZeroCount()]),
		columnIndex(new int[triplet.getNonZeroCount()]),
		rowPointer(new int[triplet.getDenseRowCount() + 1]),
		denseRowCount(triplet.getDenseRowCount()),
		denseColCount(triplet.getDenseColCount()),
		firstActiveRow(-1)
	{
		// Calloc was proven to be faster than new [rows]()
		std::unique_ptr<int[], decltype(&std::free)> rowCount(static_cast<int*>(calloc(denseRowCount, sizeof(int))), &std::free);
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
		for (int i = 0; i < denseRowCount; ++i) {
			rowPointer[i + 1] = rowCount[i] + rowPointer[i];
			if (firstActiveRow == -1 && rowCount[i] > 0) {
				firstActiveRow = i;
			}
		}
		rowPointer[denseRowCount] = triplet.getNonZeroCount();
		if (firstActiveRow == -1) {
			firstActiveRow = getDenseRowCount();
		}

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

	inline CSRConstIterator CSRMatrix::begin() const {
		return CSRConstIterator(values.get(), columnIndex.get(), rowPointer.get(), firstActiveRow, 0);
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

	/// @brief Function to convert matrix from compressed sparse row format to dense row major matrix
	/// Out must be allocated and filled with zero before being passed to the function
	/// Out will be contain linearized dense version of the matrix, first CSRMatrix::getDenseRowCount elements
	/// will be the first row of the dense matrix, next CSRMatrix::getDenseRowCount will be the second row and so on.
	/// @param[in] compressed Matrix in Compressed Sparse Row format
	/// @param[out] out Preallocated (and filled with zero) space where the dense matrix will be added
	template<typename CompressedMatrixFormat>
	static void toLinearDenseRowMajor(const CompressedMatrixFormat& compressed, float* out) noexcept {
		const int64_t colCount = compressed.getDenseColCount();
		for (const auto& el : compressed) {
			const int64_t index = el.getRow() * colCount + el.getCol();
			out[index] = el.getValue();
		}
	}
}