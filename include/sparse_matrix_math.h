#pragma once
#include <vector>
#include <unordered_map>
#include <memory>
#include <cassert>
#include <utility>
#include <cinttypes>
#include <cmath>

#ifdef SMM_MULTITHREADING_CPPTM
#include <cpp_tm.h>
#endif // SMM_MULTITHREADING_CPPTM


namespace SMM {
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

	class CSRElement {
	public:
		const float getValue() const noexcept;
		const int getRow() const noexcept;
		const int getCol() const noexcept;
		friend void swap(CSRElement& a, CSRElement& b) noexcept;
		friend class CSRConstIterator;
	protected:
		CSRElement(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;

		CSRElement(const CSRElement&) = default;
		CSRElement& operator=(const CSRElement&) = default;
		const bool operator==(const CSRElement&) const;

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

	CSRElement::CSRElement(
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

	const float CSRElement::getValue() const noexcept {
		return values[currentPositionIndex];
	}

	const int CSRElement::getRow() const noexcept {
		return currentStartIndex;
	}
	const int CSRElement::getCol() const noexcept {
		return positions[currentPositionIndex];
	}

	void swap(CSRElement& a, CSRElement& b) noexcept {
		using std::swap;
		swap(a.values, b.values);
		swap(a.positions, b.positions);
		swap(a.start, b.start);
		swap(a.currentStartIndex, b.currentStartIndex);
		swap(a.currentPositionIndex, b.currentPositionIndex);
	}

	const bool CSRElement::operator==(const CSRElement& other) const {
		return other.values == values &&
			other.positions == positions &&
			other.start == start &&
			other.currentStartIndex == currentStartIndex &&
			other.currentPositionIndex == currentPositionIndex;
	}

	/// @brief Base class for const forward iterator for matrix in compressed sparse row format
	class CSRConstIterator {
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = CSRElement;
		using pointer = const CSRElement*;
		using reference = const CSRElement&;

		CSRConstIterator() = default;
		CSRConstIterator(
			const float* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;
		CSRConstIterator(const CSRConstIterator&) = default;
		CSRConstIterator& operator=(const CSRConstIterator&) = default;
		const bool operator==(const CSRConstIterator& other) const noexcept;
		const bool operator!=(const CSRConstIterator& other) const noexcept;
		reference operator*() const;
		pointer operator->() const;
		CSRConstIterator& operator++() noexcept;
		CSRConstIterator operator++(int) noexcept;
		friend void swap(CSRConstIterator& a, CSRConstIterator& b) noexcept;
	private:
		CSRElement currentElement;
	};

	CSRConstIterator::CSRConstIterator(
		const float* values,
		const int* positions,
		const int* start,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		currentElement(values, positions, start, currentStartIndex, currentPositionIndex)
	{ }

	inline typename CSRConstIterator::reference CSRConstIterator::operator*() const {
		return currentElement;
	}

	inline typename CSRConstIterator::pointer CSRConstIterator::operator->() const {
		return &currentElement;
	}

	inline const bool CSRConstIterator::operator==(const CSRConstIterator& other) const noexcept {
		return currentElement == other.currentElement;
	}

	inline const bool CSRConstIterator::operator!=(const CSRConstIterator& other) const noexcept {
		return !(*this == other);
	}

	CSRConstIterator& CSRConstIterator::operator++() noexcept {
		currentElement.currentPositionIndex++;
		assert(currentElement.currentPositionIndex <= currentElement.start[currentElement.currentStartIndex + 1]);
		if (currentElement.currentPositionIndex == currentElement.start[currentElement.currentStartIndex + 1]) {
			do {
				currentElement.currentStartIndex++;
			} while (currentElement.start[currentElement.currentStartIndex + 1] == currentElement.currentPositionIndex);
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

	/// @brief Base class for matrix in compressed sparse row format
	/// Compressed sparse row format is represented with 3 arrays. One for the values,
	/// one to keep track nonzero columns for each row, one to keep track where each row
	class CSRMatrix {
	public:
		using ConstIterator = CSRConstIterator;
		CSRMatrix() noexcept;
		CSRMatrix(const TripletMatrix& triplet) noexcept;
		CSRMatrix(const CSRMatrix&) = delete;
		CSRMatrix& operator=(const CSRMatrix&) = delete;
		CSRMatrix(CSRMatrix&&) noexcept = default;
		CSRMatrix& operator=(CSRMatrix&&) noexcept = default;
		~CSRMatrix() = default;
		const int init(const TripletMatrix& triplet) noexcept;
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
		ConstIterator begin() const noexcept;
		ConstIterator end() const noexcept;
		void rMult(const float* const mult, float* const out, const bool async = false) const noexcept;
		void rMultAdd(const float* const lhs, const float* const mult, float* const out, const bool async = false) const noexcept;
		void rMultSub(const float* const lhs, const float* const mult, float* const out, const bool async = false) const noexcept;
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
		const int fillArrays(const TripletMatrix& triplet) noexcept;
		const int getNextStartIndex(int currentStartIndex, int startLength) const noexcept;
		template<typename FunctorType>
		void rMultOp(
			const float* const lhs,
			const float* const mult,
			float* const out,
			const FunctorType& op,
			const bool async
		) const noexcept {
#ifdef SMM_MULTITHREADING_CPPTM
			struct rMultOpTask final : public CPPTM::ITask {
				rMultOpTask(
					const CSRMatrix* csr,
					const float* const lhs,
					const float* const mult,
					float* const out,
					const FunctorType& op
				) :
					csr(csr),
					lhs(lhs),
					mult(mult),
					out(out),
					op(op)
				{ }

				const CSRMatrix* csr;
				const float* const lhs;
				const float* const mult;
				float* const out;
				const FunctorType& op;

				CPPTM::CPPTMStatus runTask(int blockIndex, int numBlocks) override {
					const int rows = csr->getDenseRowCount();
					const int blockSize = (rows + numBlocks) / numBlocks;
					const int startIdx = blockSize * blockIndex;
					const int end = std::min(rows, startIdx + blockSize);
					const int start = startIdx == 0 ? csr->firstActiveStart : csr->getNextStartIndex(startIdx - 1, rows);
					for (int row = start; row < end; row = csr->getNextStartIndex(row, rows)) {
						float dot = 0.0f;
						for (int colIdx = csr->start[row]; colIdx < csr->start[row + 1]; ++colIdx) {
							const int col = csr->positions[colIdx];
							const float val = csr->values[colIdx];
							dot = std::fmaf(val, mult[col], dot);
						}
						out[row] = op(lhs[row], dot);
					}
					return CPPTM::CPPTMStatus::SUCCESS;
				}
			};
			CPPTM::ThreadManager& globalTm = CPPTM::getGlobalTM();
			if (async) {
				std::shared_ptr taskPtr = std::make_shared<rMultOpTask>(this, lhs, mult, out, op);
				globalTm.launchAsync(taskPtr);
			} else {
				rMultOpTask task(this, lhs, mult, out, op);
				globalTm.launchSync(task);
			}
#else
			for (int row = firstActiveStart; row < denseRowCount; row = getNextStartIndex(row, denseRowCount)) {
				float dot = 0.0f;
				for (int colIdx = start[row]; colIdx < start[row + 1]; ++colIdx) {
					const int col = positions[colIdx];
					const float val = values[colIdx];
					dot = std::fmaf(val, mult[col], dot);
				}
				out[row] = op(lhs[row], dot);
			}
#endif
		}
	};

	CSRMatrix::CSRMatrix() noexcept :
		values(nullptr),
		positions(nullptr),
		start(nullptr),
		denseRowCount(0),
		denseColCount(0),
		firstActiveStart(-1)
	{ }

	CSRMatrix::CSRMatrix(const TripletMatrix& triplet) noexcept :
		values(new float[triplet.getNonZeroCount()]),
		positions(new int[triplet.getNonZeroCount()]),
		start(new int[triplet.getDenseRowCount() + 1]),
		denseRowCount(triplet.getDenseRowCount()),
		denseColCount(triplet.getDenseColCount()),
		firstActiveStart(-1)
	{
		fillArrays(triplet);
	}

	const int CSRMatrix::init(const TripletMatrix& triplet) noexcept {
		denseRowCount = triplet.getDenseRowCount();
		denseColCount = triplet.getDenseColCount();
		const int nnz = triplet.getNonZeroCount();
		values.reset(new float[nnz]);
		if (!values) {
			return 1;
		}
		positions.reset(new int[nnz]);
		if (!positions) {
			return 1;
		}
		start.reset(new int[denseRowCount + 1]);
		if (!start) {
			return 1;
		}
		int err = fillArrays(triplet);
		if (err) {
			return err;
		}
		return 0;
	}

	inline const int CSRMatrix::getNonZeroCount() const noexcept {
		return start[getDenseRowCount()];
	}

	inline const int CSRMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline const int CSRMatrix::getDenseColCount() const noexcept {
		return denseColCount;
	}

	inline CSRMatrix::ConstIterator CSRMatrix::begin() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), firstActiveStart, 0);
	}

	inline CSRMatrix::ConstIterator CSRMatrix::end() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), denseRowCount, start[denseRowCount]);
	}

	inline void CSRMatrix::rMult(const float* const mult, float* const res, const bool async) const noexcept {
		auto rhsId = [](const float lhs, const  float rhs) -> float {
			return rhs;
		};
		rMultOp(res, mult, res, rhsId, async);
	}

	inline void CSRMatrix::rMultAdd(const float* const lhs, const float* const mult, float* const out, const bool async) const noexcept {
		auto addOp = [](const float lhs, const float rhs) ->float {
			return lhs + rhs;
		};
		rMultOp(lhs, mult, out, addOp, async);
	}


	inline void CSRMatrix::rMultSub(const float* const lhs, const float* const mult, float* const out, const bool async) const noexcept {
		auto addOp = [](const float lhs, const float rhs) ->float {
			return lhs - rhs;
		};
		rMultOp(lhs, mult, out, addOp, async);
	}

	inline const int CSRMatrix::getNextStartIndex(int currentStartIndex, int startLength) const noexcept {
		do {
			currentStartIndex++;
		} while (currentStartIndex < startLength && start[currentStartIndex] == start[currentStartIndex + 1]);
		return currentStartIndex;
	}

	const int CSRMatrix::fillArrays(const TripletMatrix& triplet) noexcept {
		const int n = getDenseRowCount();
		std::unique_ptr<int[], decltype(&free)> count(static_cast<int*>(calloc(n, sizeof(int))), &free);
		if (count == nullptr) {
			assert(false);
			return 1;
		}

		for (const auto& el : triplet) {
			count[el.getRow()]++;
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
			const int startIdx = el.getRow();
			const int currentCount = count[startIdx];;
			const int position = start[startIdx + 1] - currentCount;
			positions[position] = el.getCol();
			values[position] = el.getValue();
			count[startIdx]--;
		}
		return 0;
	}

	/// @brief Function to convert matrix from compressed sparse row format to dense row major matrix
	/// Out must be allocated and filled with zero before being passed to the function
	/// Out will be contain linearized dense version of the matrix, first CSRMatrix::getDenseRowCount elements
	/// will be the first row of the dense matrix, next CSRMatrix::getDenseRowCount will be the second row and so on.
	/// @param[in] compressed Matrix in Compressed Sparse Row format
	/// @param[out] out Preallocated (and filled with zero) space where the dense matrix will be added
	template<typename CompressedMatrixFormat>
	inline void toLinearDenseRowMajor(const CompressedMatrixFormat& compressed, float* out) noexcept {
		const int64_t colCount = compressed.getDenseColCount();
		for (const auto& el : compressed) {
			const int64_t index = el.getRow() * colCount + el.getCol();
			out[index] = el.getValue();
		}
	}

	class Vector {
	public:
		Vector() noexcept :
			size(0),
			data(nullptr)
		{ }

		Vector(const int size) noexcept :
			size(size),
			data(static_cast<float*>(malloc(size * sizeof(float))))
		{ }

		Vector(const int size, const float val) noexcept :
			size(size)
		{
			initDataWithVal(val);
		}

		Vector(const std::initializer_list<float>& l) :
			size(l.size()),
			data(static_cast<float*>(malloc(l.size() * sizeof(float))))
		{
			std::copy(l.begin(), l.end(), data);
		}

		Vector(Vector&& other) noexcept :
			size(other.size) {
			free(data);
			data = other.data;
			other.data = nullptr;
			other.size = 0;
		}

		Vector& operator=(Vector&& other) noexcept {
			size = other.size;
			free(data);
			data = other.data;
			other.data = nullptr;
			other.size = 0;
			return *this;
		}

		Vector(const Vector&) = delete;
		Vector& operator=(const Vector&) = delete;

		~Vector() {
			free(data);
			data = nullptr;
			size = 0;
		}

		void init(const int size) {
			assert(this->size == 0 && data == nullptr);
			this->size = size;
			data = static_cast<float*>(malloc(size * sizeof(float)));
		}

		void init(const int size, const float val) {
			assert(this->size == 0 && data == nullptr);
			this->size = size;
			initDataWithVal(val);
		}

		const int getSize() const {
			return this->size;
		}

		operator float* const() {
			return data;
		}

		const float operator[](const int index) const {
			assert(index < size);
			return data[index];
		}

		float& operator[](const int index) {
			assert(index < size);
			return data[index];
		}

		Vector& operator+=(const Vector& other) {
			assert(other.size == size);
			for (int i = 0; i < size; ++i) {
				data[i] += other[i];
			}
			return *this;
		}

		Vector& operator-=(const Vector& other) {
			assert(other.size == size);
			for (int i = 0; i < size; ++i) {
				data[i] -= other[i];
			}
			return *this;
		}

		const float secondNorm() const {
			float sum = 0.0f;
			for (int i = 0; i < size; ++i) {
				sum += data[i] * data[i];
			}
			return std::sqrt(sum);
		}

		const float operator*(const Vector& other) const {
			assert(other.size == size);
			float dot = 0.0f;
			for (int i = 0; i < size; ++i) {
				dot = std::fmaf(data[i], other[i], dot);
			}
			return dot;
		}
	private:
		void initDataWithVal(const float val) {
			if (val == 0.0f) {
				data = static_cast<float*>(calloc(size, sizeof(float)));
			} else {
				const int64_t byteSize = int64_t(size) * sizeof(float);
				data = static_cast<float*>(malloc(byteSize));
				if (!data) return;
				for (int i = 0; i < this->size; ++i) {
					data[i] = val;
				}
			}
		}
		float* data;
		int size;
	};

	/// @brief solve a.x=b using BiConjugate gradient matrix, where matrix a is symmetric
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too 
	/// @return 0 on success, != 0 on error
	inline int BiCGSymmetric(const CSRMatrix& a, Vector& b, Vector& x) {
		auto multAddVector = [](const Vector& lhs, const Vector& rhs, const float scalar, Vector& out) {
			assert((lhs.getSize() == rhs.getSize()) && (rhs.getSize() == out.getSize()));
			for (int i = 0; i < lhs.getSize(); ++i) {
				out[i] = std::fmaf(scalar, rhs[i], lhs[i]);
			}
		};

		const float eps = 1e-4;
		int numIterations = 100;
		Vector r(b.getSize());
		a.rMultSub(b, x, r);

		Vector p(b.getSize());
		for (int i = 0; i < p.getSize(); ++i) {
			p[i] = r[i];
		}

		Vector ap(b.getSize());

		float rSquare = r * r;
		do {
			a.rMult(p, ap);
			const float denom = ap* p;
			// Numerical instability will cause devision by zero (or something close to). The method must be restarted
			if (eps > std::abs(denom)) {
				return 1;
			}
			const float alpha = rSquare / (ap * p);
			multAddVector(x, p, alpha, x);
			multAddVector(r, ap, -alpha, r);
			// Dot product r * r can be zero (or close to zero) only if r has length close to zero.
			// But if the residual is close to zero, this means that we have found a solution
			const float newRSquare = r * r;
			if (eps > newRSquare) {
				break;
			}
			const float beta = newRSquare / rSquare;
			multAddVector(r, p, beta, p);
			rSquare = newRSquare;
			numIterations--;
		} while (r.secondNorm() > eps && numIterations > 0);

		if (numIterations <= 0) {
			return 1;
		}
		return 0;
	}
}