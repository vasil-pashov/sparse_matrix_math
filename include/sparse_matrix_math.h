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

	enum CSFormat {
		CSR, ///< (C)ompressed (S)parse (R)ow
		CSC ///< (C)ompressed (S)parse (C)olumn
	};

	/// @brief Const forward iterator for matrix in compressed sparse row format
	template<CSFormat format>
	class CSConstIterator {
	private:
		class CSElement {
		friend class CSConstIterator;
		public:
			const int getRow() const noexcept;
			const int getCol() const noexcept;
			const float getValue() const noexcept;
			friend void swap(CSElement& a, CSElement& b) noexcept;
		private:
			CSElement(
				const float* values,
				const int* columnIndex,
				const int* rowPointer,
				const int currentRow,
				const int currentColumIndex
			) noexcept;
			const bool operator==(const CSElement& other) const noexcept;

			CSElement(const CSElement&) = default;
			CSElement& operator=(const CSElement&) = default;

			const float* values;
			const int* columnIndex;
			const int* rowPointer;
			int currentRow;
			int currentColumnIndex;
		};

		friend void swap(CSConstIterator<format>::CSElement& a, CSConstIterator<format>::CSElement& b) noexcept {
			using std::swap;
			swap(a.values, b.values);
			swap(a.columnIndex, b.columnIndex);
			swap(a.rowPointer, b.rowPointer);
			swap(a.currentRow, b.currentRow);
			swap(a.currentColumnIndex, b.currentColumnIndex);
		}

	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = CSConstIterator::CSElement;
		using pointer = const CSConstIterator::CSElement*;
		using reference = const CSConstIterator::CSElement&;

		CSConstIterator() = default;
		CSConstIterator(
			const float* values,
			const int* columnPointer,
			const int* rowPointer,
			const int currentRow,
			const int currentColumnIndex
		) noexcept;
		CSConstIterator(const CSConstIterator&) = default;
		CSConstIterator& operator=(const CSConstIterator&) = default;
		reference operator*() const;
		pointer operator->() const;
		const bool operator==(const CSConstIterator& other) const noexcept;
		const bool operator!=(const CSConstIterator& other) const noexcept;
		CSConstIterator& operator++() noexcept;
		CSConstIterator operator++(int) noexcept;
		friend void swap(CSConstIterator& a, CSConstIterator& b) noexcept;
	private:
		CSElement currentElement;
	};

	template<CSFormat format>
	CSConstIterator<format>::CSElement::CSElement(
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

	template<CSFormat format>
	const int CSConstIterator<format>::CSElement::getRow() const noexcept {
		return currentRow;
	}

	template<CSFormat format>
	const int CSConstIterator<format>::CSElement::getCol() const noexcept {
		return columnIndex[currentColumnIndex];
	}

	template<CSFormat format>
	const float CSConstIterator<format>::CSElement::getValue() const noexcept {
		return values[currentColumnIndex];
	}

	template<CSFormat format>
	const bool CSConstIterator<format>::CSElement::operator==(const CSElement& other) const noexcept {
		return values == other.values &&
			columnIndex == other.columnIndex &&
			rowPointer == other.rowPointer &&
			currentRow == other.currentRow &&
			currentColumnIndex == other.currentColumnIndex;
	}

	template<CSFormat format>
	CSConstIterator<format>::CSConstIterator(
		const float* values,
		const int* columnIndex,
		const int* rowPointer,
		const int currentRow,
		const int currentColumIndex
	) noexcept :
		currentElement(values, columnIndex, rowPointer, currentRow, currentColumIndex)
	{ }

	template<CSFormat format>
	inline typename CSConstIterator<format>::reference CSConstIterator<format>::operator*() const {
		return currentElement;
	}

	template<CSFormat format>
	inline typename CSConstIterator<format>::pointer CSConstIterator<format>::operator->() const {
		return &currentElement;
	}

	template<CSFormat format>
	inline const bool CSConstIterator<format>::operator==(const CSConstIterator& other) const noexcept {
		return currentElement == other.currentElement;
	}

	template<CSFormat format>
	inline const bool CSConstIterator<format>::operator!=(const CSConstIterator& other) const noexcept {
		return !(*this == other);
	}

	template<CSFormat format>
	CSConstIterator<format>& CSConstIterator<format>::operator++() noexcept {
		currentElement.currentColumnIndex++;
		assert(currentElement.currentColumnIndex <= currentElement.rowPointer[currentElement.currentRow + 1]);
		if (currentElement.currentColumnIndex == currentElement.rowPointer[currentElement.currentRow + 1]) {
			do {
				currentElement.currentRow++;
			} while (currentElement.rowPointer[currentElement.currentRow + 1] == currentElement.currentColumnIndex);
		}
		return *this;
	}

	template<CSFormat format>
	CSConstIterator<format> CSConstIterator<format>::operator++(int) noexcept {
		CSConstIterator initialState = *this;
		++(*this);
		return initialState;
	}

	template<CSFormat format>
	inline void swap(CSConstIterator<format>& a, CSConstIterator<format>& b) noexcept {
		swap(a.currentElement, b.currentElement);
	}

	/// @brief Matrix in compressed sparse row format
	/// Compressed sparse row format is represented with 3 arrays. One for the values,
	/// one to keep track nonzero columns for each row, one to keep track where each row starts
	/// The columns in each row are not ordered in any particular way (i.e. in ascending order)
	template<CSFormat format>
	class CSMatrix {
	public:
		using ConstIterator = CSConstIterator<format>;
		/// @brief Initialize matrix in compressed sparse row format from a given matrix into triplet format
		/// In case that the constructor could not allocate the needed amount of memory,
		/// denseRowCount and denseColCount are set to -1 and all memory that was allocated will be freed.
		/// @param[in] tripletMatrix Matrix in triplet format from which CSR matrix will be created
		explicit CSMatrix(const TripletMatrix& tripletMatrix) noexcept;
		CSMatrix(CSMatrix&&) noexcept;
		CSMatrix& operator=(CSMatrix&&) noexcept;
		CSMatrix(const CSMatrix&) = delete;
		CSMatrix& operator=(const CSMatrix&) = delete;
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
		ConstIterator begin() const;
		/// @brief Iterator to one element past the last element in the matrix
		/// Dereferencing this iterator results in undefined behavior.
		ConstIterator end() const;
	private:
		/// Array which will hold all nonzero entries of the matrix.
		/// This is of length  number of nonzero entries
		std::unique_ptr<float[]> values;
		/// If format is CSR this is the column if the i-th value
		/// If format is CSC this is the row of the i-th value
		std::unique_ptr<int[]> positions;
		/// If format is CSR i-th element is index in positions and values where the i-th row starts
		/// If format is CSC i-th element is index in positions and values where the i-th column starts
		std::unique_ptr<int[]> start;
		int denseRowCount; ///< Number of rows in the matrix
		int denseColCount; ///< Number of columns in the matrix
		/// Index in start array. If format is CSR the first row which has nonzero element in it
		/// If format is CSC the first column which has nonzero element in it
		int firstActiveStart;
		/// Get the length of the start array as it depends on the CSFormat of the matrix
		/// @return Length of the start array
		const int getStartLength() const;
	};

	template<CSFormat format>
	CSMatrix<format>::CSMatrix(const TripletMatrix& triplet) noexcept :
		values(new float[triplet.getNonZeroCount()]),
		positions(new int[triplet.getNonZeroCount()]),
		start(new int[(format == CSR ? triplet.getDenseRowCount() : triplet.getDenseColCount()) + 1]),
		denseRowCount(triplet.getDenseRowCount()),
		denseColCount(triplet.getDenseColCount()),
		firstActiveStart(-1)
	{
		static_assert(format == CSFormat::CSC || format == CSFormat::CSR, "Undefined matrix format");
		const int n = getStartLength();
		// Calloc was proven to be faster than new [rows]()
		// The count of elements in the i-th row/column depending of the format (CSR/CSC)
		std::unique_ptr<int[], decltype(&std::free)> count(static_cast<int*>(calloc(n, sizeof(int))), &std::free);
		if (values == nullptr || positions == nullptr || start == nullptr || count == nullptr) {
			assert(false);
			denseRowCount = -1;
			denseColCount = -1;
			values.reset();
			positions.reset();
			start.reset();
			count.reset();
			return;
		}

		for (const auto& el : triplet) {
			if (format == CSFormat::CSR) {
				count[el.getRow()]++;
			} else if (format == CSFormat::CSC) {
				count[el.getCol()]++;
			}
		}

		start[0] = 0;
		for (int i = 0; i < n; ++i) {
			start[i + 1] = count[i] + start[i];
			if (firstActiveStart == -1 && count[i] > 0) {
				firstActiveStart = i;
			}
		}
		start[n] = triplet.getNonZeroCount();
		if (firstActiveStart == -1) {
			firstActiveStart = n;
		}

		for (const auto& el : triplet) {
			const int startIndex = format == CSR ? el.getRow() : el.getCol();
			const int index = start[startIndex + 1] - count[startIndex];
			values[index] = el.getValue();
			positions[index] = format == CSR ? el.getCol() : el.getRow();
			count[startIndex]--;
		}
	}

	template<CSFormat format>
	CSMatrix<format>::CSMatrix(CSMatrix&& other) noexcept :
		values(std::move(other.values)),
		positions(std::move(other.positions)),
		start(std::move(other.start)),
		denseRowCount(other.denseRowCount),
		denseColCount(other.denseColCount)
	{
		other.denseRowCount = 0;
		other.denseColCount = 0;
	}

	template<CSFormat format>
	CSMatrix<format>& CSMatrix<format>::operator=(CSMatrix&& other) noexcept {
		values = std::move(other.values);
		positions = std::move(other.positions);
		start = std::move(other.start);
		denseRowCount = other.denseRowCount;
		denseColCount = other.denseColCount;
		other.denseRowCount = 0;
		other.denseColCount = 0;
		return *this;
	}

	template<CSFormat format>
	inline const int CSMatrix<format>::getNonZeroCount() const noexcept {
		return start[denseRowCount];
	}

	template<CSFormat format>
	inline const int CSMatrix<format>::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	template<CSFormat format>
	inline const int CSMatrix<format>::getDenseColCount() const noexcept {
		return denseColCount;
	}

	template<CSFormat format>
	inline typename CSMatrix<format>::ConstIterator CSMatrix<format>::begin() const {
		return ConstIterator(values.get(), positions.get(), start.get(), firstActiveStart, 0);
	}

	template<CSFormat format>
	inline typename CSMatrix<format>::ConstIterator CSMatrix<format>::end() const {
		const int nonZeroCount = getNonZeroCount();
		return ConstIterator(
			values.get(),
			positions.get(),
			start.get(),
			getStartLength(),
			nonZeroCount
		);
	}

	template<CSFormat format>
	inline const int CSMatrix<format>::getStartLength() const {
		return format == CSR ? denseRowCount: denseColCount;
	}

	using CSRMatrix = CSMatrix<CSFormat::CSR>;
	using CSCMatrix = CSMatrix<CSFormat::CSC>;

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