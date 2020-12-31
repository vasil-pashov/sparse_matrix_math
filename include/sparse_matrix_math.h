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
		class TripletEl {
		public:
			friend class TripletMatrix;
			TripletEl(int row, int col, float value) noexcept :
				row(row),
				col(col),
				value(value) {
			}
			TripletEl(const TripletEl&) noexcept = default;
			TripletEl(TripletEl&&) noexcept = default;
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
		using ConstIterator = std::vector<TripletEl>::const_iterator;
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
		std::vector<TripletEl> data;
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

	class CSElement {
	public:
		const float getValue() const noexcept;
		friend void swap(CSElement& a, CSElement& b) noexcept;
	protected:
		CSElement(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;

		CSElement(const CSElement&) = default;
		CSElement& operator=(const CSElement&) = default;
		const bool operator==(const CSElement&) const;

		/// Non owning pointer to the list of non zero elements of the matrix
		const float* values;
		/// If format is CSR this is the column if the i-th value
		/// If format is CSC this is the row of the i-th value
		const int* positions;
		/// If format is CSR i-th element is index in positions and values where the i-th row starts
		/// If format is CSC i-th element is index in positions and values where the i-th column starts
		const int* start;
		/// Index into start for the element which the iterator is pointing to
		int currentStartIndex;
		/// Index into positions for the element which the iterator is pointing to
		int currentPositionIndex;
	};

	CSElement::CSElement(
		const float* values,
		const int* positions,
		const int* start,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		values(values),
		positions(positions),
		start(start),
		currentStartIndex(currentStartIndex),
		currentPositionIndex(currentPositionIndex) {
	}

	const float CSElement::getValue() const noexcept {
		return values[currentPositionIndex];
	}

	void swap(CSElement& a, CSElement& b) noexcept {
		using std::swap;
		swap(a.values, b.values);
		swap(a.positions, b.positions);
		swap(a.start, b.start);
		swap(a.currentStartIndex, b.currentStartIndex);
		swap(a.currentPositionIndex, b.currentPositionIndex);
	}

	const bool CSElement::operator==(const CSElement& other) const {
		return other.values == values &&
			other.positions == positions &&
			other.start == start &&
			other.currentStartIndex == currentStartIndex &&
			other.currentPositionIndex == currentPositionIndex;
	}

	/// @brief Base class for const forward iterator for matrix in compressed sparse row format
	template<typename CSElement>
	class CSConstIterator {
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = CSElement;
		using pointer = const CSElement*;
		using reference = const CSElement&;

		CSConstIterator() = default;
		CSConstIterator(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;
		CSConstIterator(const CSConstIterator&) = default;
		CSConstIterator& operator=(const CSConstIterator&) = default;
		const bool operator==(const CSConstIterator& other) const noexcept;
		const bool operator!=(const CSConstIterator& other) const noexcept;
		reference operator*() const;
		pointer operator->() const;
		CSConstIterator& operator++() noexcept;
		CSConstIterator operator++(int) noexcept;
		friend void swap(CSConstIterator& a, CSConstIterator& b) noexcept;
	private:
		CSElement currentElement;
	};

	template<typename CSElement>
	CSConstIterator<CSElement>::CSConstIterator(
		const float* values,
		const int* positions,
		const int* start,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		currentElement(values, positions, start, currentStartIndex, currentPositionIndex)
	{ }

	template<typename CSElement>
	inline typename CSConstIterator<CSElement>::reference CSConstIterator<CSElement>::operator*() const {
		return currentElement;
	}

	template<typename CSElement>
	inline typename CSConstIterator<CSElement>::pointer CSConstIterator<CSElement>::operator->() const {
		return &currentElement;
	}

	template<typename CSElement>
	inline const bool CSConstIterator<CSElement>::operator==(const CSConstIterator& other) const noexcept {
		return currentElement == other.currentElement;
	}

	template<typename CSElement>
	inline const bool CSConstIterator<CSElement>::operator!=(const CSConstIterator& other) const noexcept {
		return !(*this == other);
	}

	template<typename CSElement>
	CSConstIterator<CSElement>& CSConstIterator<CSElement>::operator++() noexcept {
		currentElement.currentPositionIndex++;
		assert(currentElement.currentPositionIndex <= currentElement.start[currentElement.currentStartIndex + 1]);
		if (currentElement.currentPositionIndex == currentElement.start[currentElement.currentStartIndex + 1]) {
			do {
				currentElement.currentStartIndex++;
			} while (currentElement.start[currentElement.currentStartIndex + 1] == currentElement.currentPositionIndex);
		}
		return *this;
	}

	template<typename CSElement>
	CSConstIterator<CSElement> CSConstIterator<CSElement>::operator++(int) noexcept {
		CSConstIterator initialState = *this;
		++(*this);
		return initialState;
	}

	template<typename CSElement>
	inline void swap(CSConstIterator<CSElement>& a, CSConstIterator<CSElement>& b) noexcept {
		swap(a.currentElement, b.currentElement);
	}

	class CSRElement : public CSElement {
		friend class CSConstIterator<CSRElement>;
	public:
		CSRElement(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;
		const int getRow() const noexcept {
			return currentStartIndex;
		}

		const int getCol() const noexcept {
			return positions[currentPositionIndex];
		}
	};

	CSRElement::CSRElement(
		const float* values,
		const int* positions,
		const int* start,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		CSElement(values, positions, start, currentStartIndex, currentPositionIndex)
	{ }

	class CSCElement : public CSElement {
		friend class CSConstIterator<CSCElement>;
	public:
		CSCElement(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;
		const int getRow() const noexcept {
			return positions[currentPositionIndex];
		}

		const int getCol() const noexcept {
			return currentStartIndex;
		}
	};

	CSCElement::CSCElement(
		const float* values,
		const int* positions,
		const int* start,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		CSElement(values, positions, start, currentStartIndex, currentPositionIndex) 			{
 }

	/// @brief Base class for matrix in compressed sparse row format
	/// Compressed sparse row format is represented with 3 arrays. One for the values,
	/// one to keep track nonzero columns/rows for each row/column, one to keep track where each row/column starts
	class CSMatrix {
	public:
		CSMatrix(const CSMatrix&) = delete;
		CSMatrix& operator=(const CSMatrix&) = delete;
		/// @brief Get the total number of rows of the matrix
		/// @return The row count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseRowCount() const noexcept;
		/// @brief Get the total number of cols of the matrix
		/// @return The column count which dense matrix is supposed to have (not only the stored ones)
		const int getDenseColCount() const noexcept;
	protected:
		CSMatrix() noexcept;
		/// May fail due to insufficient memory
		CSMatrix(
			const int nonZeroCount,
			const int startSize,
			const int denseRowCount,
			const int denseColumnCount
		) noexcept;
		CSMatrix(CSMatrix&&) noexcept = default;
		CSMatrix& operator=(CSMatrix&&) noexcept = default;
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
		/// @brief Release all allocated memory and set all counts to 0
		void freeMem() noexcept;
		const int init(
			const int nonZeroCount,
			const int startSize,
			const int denseRowCount,
			const int denseColumnCount
		) noexcept;
		
		enum type {
			CSR,
			CSC
		};

		template<type t>
		void fillArrays(const TripletMatrix& triplet) {
			const int n = t == CSR ? getDenseRowCount() : getDenseColCount();
			std::unique_ptr<int[], decltype(&free)> count(static_cast<int*>(calloc(n, sizeof(int))), &free);
			if (count == nullptr) {
				assert(false);
				freeMem();
			}

			for (const auto& el : triplet) {
				if (t == CSR) {
					count[el.getRow()]++;
				} else {
					count[el.getCol()]++;
				}
			}

			start[0] = 0;
			for (int i = 0; i < n; ++i) {
				start[i + 1] = start[i] + count[i];
				if (firstActiveStart == -1 && start[i + 1] != 0) {
					firstActiveStart = i;
				}
			}
			if (firstActiveStart == -1) {
				firstActiveStart = n;
			}

			for (const auto& el : triplet) {
				const int startIdx = t == CSR ? el.getRow() : el.getCol();
				const int currentCount = count[startIdx];;
				const int position = start[startIdx + 1] - currentCount;
				positions[position] = t == CSR ? el.getCol() : el.getRow();
				values[position] = el.getValue();
				count[startIdx]--;
			}
		}

		const int getNextStartIndex(int currentStartIndex, int startLength) const noexcept;
	};

	CSMatrix::CSMatrix() noexcept :
		denseRowCount(0),
		denseColCount(0),
		firstActiveStart(0)
	{ }

	CSMatrix::CSMatrix(
		const int nonZeroCount,
		const int startSize,
		const int denseRowCount,
		const int denseColCount
	) noexcept :
		values(new float[nonZeroCount]),
		positions(new int[nonZeroCount]),
		start(new int[startSize]),
		denseRowCount(denseRowCount),
		denseColCount(denseColCount),
		firstActiveStart(-1)
	{
		if (values == nullptr || positions == nullptr || start == nullptr) {
			assert(false);
			freeMem();
		}
	}

	inline const int CSMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline const int CSMatrix::getDenseColCount() const noexcept {
		return denseColCount;
	}

	inline void CSMatrix::freeMem() noexcept {
		this->denseColCount = 0;
		this->denseRowCount = 0;
		values.reset();
		positions.reset();
		start.reset();
	}

	inline const int CSMatrix::getNextStartIndex(int currentStartIndex, int startLength) const noexcept {
		do {
			currentStartIndex++;
		} while (currentStartIndex < startLength && start[currentStartIndex] == start[currentStartIndex + 1]);
		return currentStartIndex;
	}

	const int CSMatrix::init(
		const int nonZeroCount,
		const int startSize,
		const int denseRowCount,
		const int denseColumnCount
	) noexcept {
		values.reset(new float[nonZeroCount]);
		if (!values) { return 1; }

		positions.reset(new int[nonZeroCount]);
		if (!values) { return 1; }

		start.reset(new int[startSize]);
		if (!values) { return 1; }

		this->denseRowCount = denseRowCount;
		this->denseColCount = denseColCount;
		this->firstActiveStart = -1;
		return 0;
	}

	class CSRMatrix : public CSMatrix {
	public:
		using ConstIterator = CSConstIterator<CSRElement>;
		CSRMatrix() = default;
		explicit CSRMatrix(const TripletMatrix& triplet) noexcept;
		int init(const TripletMatrix& triplet) noexcept;
		/// @brief Get the number of trivial nonzero entries in the matrix
		/// Trivial nonzero entries do not include zero elements which came from numerical cancellation
		/// @return The number of trivial nonzero entries in the matrix
		const int getNonZeroCount() const noexcept;
		/// Get constant iterator to the first nonzero element of the matrix
		/// The iterator is guaranteed to iterate over rows in non descending fashion, eventually reaching the end of the matrix
		/// There are no guarantees for the order of the columns in each row
		/// @return Iterator to the first nonzero element of the matrix
		ConstIterator begin() const noexcept;
		/// @brief Iterator denoting the end of the matrix. Dereferencing it is undefined behavior.
		/// @return Iterator denoting the end of the matrix.
		ConstIterator end() const noexcept;
		/// @brief Multiply the matrix with the first argument being on the right hand side of the matrix and add the result to the second argument
		/// The operation overrides the second argument
		/// @param[in] mult Vector which will multiply the matrix (the vector being on the rhs of the matrix)
		/// @param[in,out] add Vector which will be added to the result of the multiplication.
		void rMultAdd(const float* const mult, float* const add) const noexcept;
	};

	CSRMatrix::CSRMatrix(const TripletMatrix& triplet) noexcept:
		CSMatrix(triplet.getNonZeroCount(), triplet.getDenseRowCount() + 1, triplet.getDenseRowCount(), triplet.getDenseColCount()) 
	{
		fillArrays<CSMatrix::CSR>(triplet);
	}

	int CSRMatrix::init(const TripletMatrix& triplet) noexcept {
		const int nnz = triplet.getNonZeroCount();
		const int rows = triplet.getDenseRowCount();
		const int cols = triplet.getDenseColCount();
		int err = CSMatrix::init(nnz, rows + 1, rows, cols);
		if (err) return err;
		CSMatrix::fillArrays<CSR>(triplet);
		return 0;
	}

	CSRMatrix::ConstIterator CSRMatrix::begin() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), this->firstActiveStart, 0);
	}

	CSRMatrix::ConstIterator CSRMatrix::end() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), getDenseRowCount(), getNonZeroCount());
	}

	inline const int CSRMatrix::getNonZeroCount() const noexcept {
		return start[denseRowCount];
	}

	void CSRMatrix::rMultAdd(const float* const mult, float* const add) const noexcept {
		const int n = getDenseRowCount();
		for (int row = firstActiveStart; row < n; row = getNextStartIndex(row, n)) {
			float res = 0.0f;
			for (int colIdx = start[row]; colIdx < start[row + 1]; ++colIdx) {
				const int col = positions[colIdx];
				const float value = values[colIdx];
				res = std::fmaf(value, mult[col], res);
			}
			add[row] += res;
		}
	}

	class CSCMatrix : public CSMatrix {
	public:
		using ConstIterator = CSConstIterator<CSCElement>;
		CSCMatrix() = default;
		explicit CSCMatrix(const TripletMatrix& triplet) noexcept;
		int init(const TripletMatrix& triplet) noexcept;
		/// @brief Get the number of trivial nonzero entries in the matrix
		/// Trivial nonzero entries do not include zero elements which came from numerical cancellation
		/// @return The number of trivial nonzero entries in the matrix
		const int getNonZeroCount() const noexcept;
		/// Get constant iterator to the first nonzero element of the matrix
		/// The iterator is guaranteed to iterate over columns in non descending fashion, eventually reaching the end of the matrix
		/// There are no guarantees for the order of the rows in each column
		/// @return Iterator to the first nonzero element of the matrix
		ConstIterator begin() const noexcept;
		/// @brief Iterator denoting the end of the matrix. Dereferencing it is undefined behavior.
		/// @return Iterator denoting the end of the matrix.
		ConstIterator end() const noexcept;
		/// @brief Multiply the matrix with the first argument being on the right hand side of the matrix and add the result to the second argument
		/// The operation overrides the second argument
		/// @param[in] mult Vector which will multiply the matrix (the vector being on the rhs of the matrix)
		/// @param[in,out] add Vector which will be added to the result of the multiplication.
		void rMultAdd(const float* const mult, float* const add) const noexcept;
	};

	CSCMatrix::CSCMatrix(const TripletMatrix& triplet) noexcept :
		CSMatrix(triplet.getNonZeroCount(), triplet.getDenseColCount() + 1, triplet.getDenseRowCount(), triplet.getDenseColCount()) 
	{
		fillArrays<CSMatrix::CSC>(triplet);
	}

	CSCMatrix::ConstIterator CSCMatrix::begin() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), this->firstActiveStart, 0);
	}

	CSCMatrix::ConstIterator CSCMatrix::end() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), getDenseColCount(), getNonZeroCount());
	}

	inline const int CSCMatrix::getNonZeroCount() const noexcept {
		return start[getDenseColCount()];
	}

	int CSCMatrix::init(const TripletMatrix& triplet) noexcept {
		const int nnz = triplet.getNonZeroCount();
		const int rows = triplet.getDenseRowCount();
		const int cols = triplet.getDenseColCount();
		int err = CSMatrix::init(nnz, rows + 1, rows, cols);
		if (err) return err;
		CSMatrix::fillArrays<CSC>(triplet);
		return 0;
	}

	void CSCMatrix::rMultAdd(const float* const mult, float* const add) const noexcept {
		const int n = getDenseColCount();
		for (int col = firstActiveStart; col < n; col = getNextStartIndex(col, n)) {
			if (mult[col] == 0.0f) {
				continue;
			}
			for (int rowIdx = start[col]; rowIdx < start[col + 1]; ++rowIdx) {
				const int row = positions[rowIdx];
				const float value = values[rowIdx];
				add[row] = std::fma(value, mult[col], add[row]);
			}
		}
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