#pragma once
#include <vector>
#include <map>
#include <memory>
#include <cassert>
#include <utility>
#include <cinttypes>
#include <cmath>
#include <fstream>
#include <cctype>
#include <iomanip>
#include <cstring>
#include <algorithm>

#ifdef SMM_MULTITHREADING_CPPTM
#include <cpp_tm/cpp_tm.h>
#if CPP_TM_MAJOR_VERSION != 0 || CPP_TM_MINOR_VERSION != 2
#error "Expected verson 0.2 of cpp_tm library"
#endif
#endif // SMM_MULTITHREADING_CPPTM

#define SMM_MAJOR_VERSION 0
#define SMM_MINOR_VERSION 2
#define SMM_PATCH_VERSION 0

namespace SMM {

#ifdef SMM_DEBUG_DOUBLE
	using real = double;
#else
	using real = float;
#endif

	namespace {
		/// Perform a * x + b, based on compilerd defines this might use actual fused multiply add call
		const inline real _smm_fma(real a, real x, real b) {
			#ifdef SMM_WITH_STD_FMA
				return std::fma(a, x, b);
			#else
				return a * x + b;
			#endif
		}
	}

	class Vector {
	public:
		Vector() noexcept :
			size(0),
			data(nullptr)
		{ }

		explicit Vector(const int size) noexcept :
			size(size),
			data(static_cast<real*>(malloc(size * sizeof(real))))
		{ }

		Vector(const int size, const real val) noexcept :
			size(size)
		{
			initDataWithVal(val);
		}

		Vector(const std::initializer_list<real>& l) :
			size(l.size()),
			data(static_cast<real*>(malloc(l.size() * sizeof(real))))
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
			data = static_cast<real*>(malloc(size * sizeof(real)));
		}

		void init(const int size, const real val) {
			assert(this->size == 0 && data == nullptr);
			this->size = size;
			initDataWithVal(val);
		}

		const int getSize() const {
			return this->size;
		}

		operator real* const() {
			return data;
		}

		const real operator[](const int index) const {
			assert(index < size);
			return data[index];
		}

		real& operator[](const int index) {
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

		const real secondNorm() const {
			real sum = 0.0f;
			for (int i = 0; i < size; ++i) {
				sum += data[i] * data[i];
			}
			return std::sqrt(sum);
		}

		const real secondNormSquared() const {
			real sum = 0.0f;
			for (int i = 0; i < size; ++i) {
				sum += data[i] * data[i];
			}
			return sum;
		}

		const real operator*(const Vector& other) const {
			assert(other.size == size);
			real dot = 0.0f;
			for (int i = 0; i < size; ++i) {
				dot += other[i] * data[i];
			}
			return dot;
		}

		real* const begin() noexcept {
			return data;
		}

		real* const end() noexcept {
			return data + size;
		}

		void fill(const real value) {
			if(value == 0.0f) {
				memset(data, 0, sizeof(real) * size);
			} else {
				std::fill_n(data, size, value);
			}
		}

	private:
		void initDataWithVal(const real val) {
			if (val == 0.0f) {
				data = static_cast<real*>(calloc(size, sizeof(real)));
			} else {
				const int64_t byteSize = int64_t(size) * sizeof(real);
				data = static_cast<real*>(malloc(byteSize));
				if (!data) return;
				for (int i = 0; i < this->size; ++i) {
					data[i] = val;
				}
			}
		}
		real* data;
		int size;
	};

	class TripletEl {
	friend class TripletMatrixConstIterator;
	public:
		TripletEl(const TripletEl&) noexcept = default;
		TripletEl& operator=(const TripletEl& other) = default;

		TripletEl(TripletEl&&) noexcept = default;
		TripletEl& operator=(TripletEl&& other) = default;

		const bool operator==(const TripletEl& other) const {
			return it == other.it;
		}

		const int getRow() const noexcept {
			return it->first >> 32;
		}

		const int getCol() const noexcept {
			return it->first & 0xFFFFFFFF;
		}

		const real getValue() const noexcept {
			return it->second;
		}

		friend void swap(TripletEl& a, TripletEl& b) noexcept;
	private:
		TripletEl(std::map<uint64_t, real>::const_iterator it) : it(it) {

		}

		std::map<uint64_t, real>::const_iterator it;
	};

	inline void swap(TripletEl& a, TripletEl& b) noexcept {
		using std::swap;
		swap(a.it, b.it);
	}
	
	class TripletMatrixConstIterator {
	friend class TripletMatrix;
	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = TripletEl;
		using pointer = const TripletEl*;
		using reference = const TripletEl&;

		TripletMatrixConstIterator& operator=(const TripletMatrixConstIterator&) = default;

		const bool operator==(const TripletMatrixConstIterator& other) const noexcept {
			return currentEl == other.currentEl;
		}

		const bool operator!=(const TripletMatrixConstIterator& other) const noexcept {
			return !(*this == other);
		}

		reference operator*() const {
			return currentEl;
		}

		pointer operator->() const {
			return &currentEl;
		}

		TripletMatrixConstIterator& operator++() noexcept {
			++currentEl.it;
			return *this;
		}

		TripletMatrixConstIterator operator++(int) noexcept {
			TripletMatrixConstIterator initialState = *this;
			++(*this);
			return initialState;
		}

		friend void swap(TripletMatrixConstIterator& a, TripletMatrixConstIterator& b) noexcept;
	private:
		TripletMatrixConstIterator(std::map<uint64_t, real>::const_iterator it) : currentEl(it) {}
		TripletEl currentEl;
	};

	inline void swap(TripletMatrixConstIterator& a, TripletMatrixConstIterator& b) noexcept {
		swap(a.currentEl, b.currentEl);
	}

	/// @brief Class to hold sparse matrix into triplet (coordinate) format.
	/// Triplet format represents the matrix entries as list of triplets (row, col, value)
	/// It is allowed repetition of elements, i.e. row and col can be the same for two
	/// separate entries, when this happens elements are being summed. Repeating elements does not
	/// increase the count of non zero elements. This class is supposed to be used as intermediate class to add
	/// entries dynamically. After all data is gathered it should be converted to CSRMatrix which provides
	/// various arithmetic functions.
	class TripletMatrix {
	public:
		using ConstIterator = TripletMatrixConstIterator;
		TripletMatrix();
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
		/// @brief Initialize triplet matrix.
		/// Be sure to call this on empty matrices (either created via the default constructor or after calling deinit())
		/// @param[in] rowCount Number of rows which the dense form of the matrix is supposed to have
		/// @param[in] colCount Number of columns which the dense form of the matrix is supposed to have
		/// @param[in] numTriplets How many elements to allocate space for.
		void init(int rowCount, int colCount, int numTriplets);
		/// @brief Free all memory used by the matrix and make it look like an empty matrix
		/// Will also set the number of dense rows and cols to zero
		void deinit();
		/// @brief Add triplet entry to the matrix
		/// @param row Row of the element
		/// @param col Column of the element
		/// @param value The value of the element at (row, col)
		void addEntry(int row, int col, real value);
		/// @brief Get constant iterator to the first element of the triplet list
		/// @return Constant iterator to the first element of the triplet list
		ConstIterator begin() const noexcept;
		/// @brief Get constant iterator to the end of the triplet list
		/// @return Constant iterator to the end of the triplet list
		ConstIterator end() const noexcept;
		/// @brief Get the number of triplets
		/// @return Number of triplets into the array
		int getNonZeroCount() const noexcept;
		/// @brief Get the total number of rows of the matrix
		/// @return The row count which dense matrix is supposed to have (not only the stored ones)
		int getDenseRowCount() const noexcept;
		/// @brief Get the total number of cols of the matrix
		/// @return The column count which dense matrix is supposed to have (not only the stored ones)
		int getDenseColCount() const noexcept;
	private:
		/// @brief Keep track of repeating indexes
		/// Key is unique representation of the matrix index (first 32 bits are the col second are the row)
		/// The value is index into the data array of triplets where the element is
		std::map<uint64_t, real> data;
		int denseRowCount; ///< Number of rows in the matrix  
		int denseColCount; ///< Number of columns in the matrix
	};

	inline TripletMatrix::TripletMatrix() :
		denseRowCount(0),
		denseColCount(0)
	{ }

	inline TripletMatrix::TripletMatrix(int denseRowCount, int denseColCount) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{ }

	inline TripletMatrix::TripletMatrix(int denseRowCount, int denseColCount, int numTriplets) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{ }

	inline void TripletMatrix::init(const int denseRowCount, const int denseColCount, const int numTriplets) {
		assert(getNonZeroCount() == 0 && getDenseRowCount() == 0 && getDenseColCount() == 0);
		this->denseRowCount = denseRowCount;
		this->denseColCount = denseColCount;
	}

	inline void TripletMatrix::deinit() {
		denseRowCount = 0;
		denseColCount = 0;
		data.clear();
	}

	inline void TripletMatrix::addEntry(int row, int col, real value) {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && row < denseColCount);
		const uint64_t key = (uint64_t(row) << 32) | uint64_t(col);
		auto it = data.find(key);
		if (it == data.end()) {
			data[key] = value;
		} else {
			data[key] += value;
		}
	}

	inline TripletMatrix::ConstIterator TripletMatrix::begin() const noexcept {
		return ConstIterator(data.cbegin());
	}

	inline TripletMatrix::ConstIterator TripletMatrix::end() const noexcept {
		return ConstIterator(data.cend());
	}

	inline int TripletMatrix::getNonZeroCount() const noexcept {
		return data.size();
	}

	inline int TripletMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline int TripletMatrix::getDenseColCount() const noexcept {
		return denseColCount;
	}

	enum CSFormat {
		CSR, ///< (C)ompressed (S)parse (R)ow
		CSC ///< (C)ompressed (S)parse (C)olumn
	};

	class CSRElement {
	public:
		const real getValue() const noexcept;
		const int getRow() const noexcept;
		const int getCol() const noexcept;
		friend void swap(CSRElement& a, CSRElement& b) noexcept;
		friend class CSRConstIterator;
	protected:
		CSRElement(
			const real* values,
			const int* positions,
			const int* start,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;

		CSRElement(const CSRElement&) = default;
		CSRElement& operator=(const CSRElement&) = default;
		const bool operator==(const CSRElement&) const;

		/// Non owning pointer to the list of non zero elements of the matrix
		const real* values;
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

	inline CSRElement::CSRElement(
		const real* values,
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

	inline const real CSRElement::getValue() const noexcept {
		return values[currentPositionIndex];
	}

	inline const int CSRElement::getRow() const noexcept {
		return currentStartIndex;
	}
	inline const int CSRElement::getCol() const noexcept {
		return positions[currentPositionIndex];
	}

	inline void swap(CSRElement& a, CSRElement& b) noexcept {
		using std::swap;
		swap(a.values, b.values);
		swap(a.positions, b.positions);
		swap(a.start, b.start);
		swap(a.currentStartIndex, b.currentStartIndex);
		swap(a.currentPositionIndex, b.currentPositionIndex);
	}

	inline const bool CSRElement::operator==(const CSRElement& other) const {
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

		CSRConstIterator(
			const real* values,
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

	inline CSRConstIterator::CSRConstIterator(
		const real* values,
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

	inline CSRConstIterator& CSRConstIterator::operator++() noexcept {
		currentElement.currentPositionIndex++;
		assert(currentElement.currentPositionIndex <= currentElement.start[currentElement.currentStartIndex + 1]);
		if (currentElement.currentPositionIndex == currentElement.start[currentElement.currentStartIndex + 1]) {
			do {
				currentElement.currentStartIndex++;
			} while (currentElement.start[currentElement.currentStartIndex + 1] == currentElement.currentPositionIndex);
		}
		return *this;
	}

	inline CSRConstIterator CSRConstIterator::operator++(int) noexcept {
		CSRConstIterator initialState = *this;
		++(*this);
		return initialState;
	}

	inline void swap(CSRConstIterator& a, CSRConstIterator& b) noexcept {
		swap(a.currentElement, b.currentElement);
	}

	enum class SolverPreconditioner {
		NONE,
		SYMMETRIC_GAUS_SEIDEL,
		ILU0
	};

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

		int init(const TripletMatrix& triplet) noexcept;
		/// @brief Get the number of trivial nonzero entries in the matrix
		/// Trivial nonzero entries do not include zero elements which came from numerical cancellation
		/// @return The number of trivial nonzero entries in the matrix
		int getNonZeroCount() const noexcept;
		/// @brief Get the total number of rows of the matrix
		/// @return The row count which dense matrix is supposed to have (not only the stored ones)
		int getDenseRowCount() const noexcept;
		/// @brief Get the total number of cols of the matrix
		/// @return The column count which dense matrix is supposed to have (not only the stored ones)
		int getDenseColCount() const noexcept;
		/// @brief Get constant iterator to the beggining of the matrix.
		/// The iterator will start at the first non empty row and will finish at the last non empty row
		/// The iterator is guaranteed to iterate over rows in increasing fashion
		/// The iterator is invalidated after init is called, or the object is destructed. Uses of invalid iterators are undefined behavior
		/// @returns Constant iterator to the beggining of the matrix
		/// Check if two matrices have the same nonzero pattern
		/// @param[in] other The matrix which will be checked against the current one
		/// @returns True if the two matrices share the same nonzero pattern
		bool hasSameNonZeroPattern(const CSRMatrix& other);
		ConstIterator begin() const noexcept;
		/// @brief Iterator to one element past the end of the matrix
		/// It is undefined to dereference this iterator. Use it only in loop checks.
		ConstIterator end() const noexcept;
		/// @brief Perform matrix vector multiplication out = A * mult
		/// Where A is the current CSR matrix
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		void rMult(const real* const mult, real* const out) const noexcept;
		/// @brief Perform matrix vector multiplication and addition: out = lhs + A * mult
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector which will be added to the matrix vector product A * mult
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		///< Makes sense only of multithreading is enabaled.
		void rMultAdd(const real* const lhs, const real* const mult, real* const out) const noexcept;
		/// @brief Perform matrix vector multiplication and subtraction: out = lhs - A * mult
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector from which the matrix vector product A * mult will be subtracted
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		///< Makes sense only of multithreading is enabaled.
		void rMultSub(const real* const lhs, const real* const mult, real* const out) const noexcept;

		/// Inplace multiplication by a scalar
		/// @param[in] scalar The scalar which will multiply the matrix
		void operator*=(const real scalar);
		/// Inplace addition of two CSR matrices.
		/// @param[in] other The matrix which will be added to the current one
		void inplaceAdd(const CSRMatrix& other);
		/// Inplace subtraction of two CSR matrices.
		/// @param[in] other The matrix which will be subtracted from the current one
		void inplaceSubtract(const CSRMatrix& other);

		/// If a non zero entry exists at position (row, col) set its value
		/// Does nothing if there is no nonzero entry at (row, col)
		/// @param[in] row The row of the entry which is going to be updated
		/// @param[in] col The column of the entry which is going to be updated
		/// @param[in] newValue The new value of the entry at (row, col)
		/// @returns True if the there is an entry at (row, col) and the update is successful
		bool updateEntry(const int row, const int col, const real newValue);

		// ********************************************************************
		// *********** MULTITHREADED VARIANTS OF MATRIX FUNCTIONS *************
		// ********************************************************************
		#if defined(SMM_MULTITHREADING_CPPTM)
		/// @brief Perform matrix vector multiplication out = A * mult in a multithreaded fashion.
		/// Where A is the current CSR matrix
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in, out] tm The thred manager which will run the multithreaded job
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		void rMult(const real* const mult, real* const out, CPPTM::ThreadManager& tm, const bool async) const noexcept;
		/// @brief Perform matrix vector multiplication and addition: out = lhs + A * mult in a multithreaded fashion.
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector which will be added to the matrix vector product A * mult
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in, out] tm The thred manager which will run the multithreaded job
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		void rMultAdd(const real* const lhs, const real* const mult, real* const out, CPPTM::ThreadManager& tm, const bool async) const noexcept;
		/// @brief Perform matrix vector multiplication and subtraction: out = lhs - A * mult
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector from which the matrix vector product A * mult will be subtracted
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in, out] tm The thred manager which will run the multithreaded job
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		void rMultSub(const real* const lhs, const real* const mult, real* const out, CPPTM::ThreadManager& tm, const bool async) const noexcept;
		#endif

		// ********************************************************************
		// ************************ PRECONDITIONERS ***************************
		// ********************************************************************

		/// Identity preconditioner. Does nothing, but implements the interface
		class IDPreconditioner {
			int apply(const real* rhs, real* x) const noexcept {
				return 0;
			}
		};

		/// Symmetric Gauss-Seidel Preconditioner.
		class SGSPreconditioner {
		public:
			SGSPreconditioner(const CSRMatrix& m) noexcept;
			SGSPreconditioner(const SGSPreconditioner&) = delete;
			SGSPreconditioner& operator=(const SGSPreconditioner&) = delete;
			SGSPreconditioner(SGSPreconditioner&&) noexcept = default;
			/// Apply the Symmetric Gauss-Seidel preconditioner to a vector
			/// @param[in] rhs Vector which will be preconditioned
			/// @param[out] x The result of preconditioning rhs
			/// @retval Non zero on error
			const int apply(const real* rhs, real* x) const noexcept;
		private:
			const CSRMatrix& m;
		};

		/// Zero fill in Incomplete LU Factorization
		class ILU0Preconditioner {
		public:
			ILU0Preconditioner(const CSRMatrix& m) noexcept;
			ILU0Preconditioner(const ILU0Preconditioner&) = delete;
			ILU0Preconditioner& operator=(const ILU0Preconditioner&) = delete;
			ILU0Preconditioner(ILU0Preconditioner&&) noexcept = default;
			/// Apply the Symmetric Gauss-Seidel preconditioner to a vector
			/// @param[in] rhs Vector which will be preconditioned
			/// @param[out] x The result of preconditioning rhs
			/// @retval Non zero on error
			const int apply(const real* rhs, real* x) const noexcept;
			const int validate() noexcept;
		private:
			/// Uses the reference to the original matrix m in order to create the LU facroization
			/// The values for both L and U will be written in ilo0Val array, the ones whic usually
			/// appear on the main diagonal of L (or U) will not be written, we must keep in mind that they are there tough.
			int factorize() noexcept;
			/// Reference to the matrix for which this decomposition is made
			/// For this factorization the resulting matrix will have the same non zero pattern as the original matrix,
			/// Thus we will reuse the two arrays start and positions from the original matrix and allocate space only for the values
			const CSRMatrix& m;
			/// The values array for the ILU0 preconditioner
			std::unique_ptr<real[]> ilu0Val;
		};

		/// Factory function to generate preconditioners for this matrix
		/// @tparam precond Type of the preconditioner defined by the SolverPreconditioner enum
		/// @returns Preconditioner which can be used for this matrix
		template<SolverPreconditioner precond>
		decltype(auto) getPreconditioner() const noexcept;
	private:
		/// Array which will hold all nonzero entries of the matrix.
		/// This is of length  number of non-zero entries
		std::unique_ptr<real[]> values;
		/// This is the column of the i-th value
		/// Sort the columns in increasing fasion. Some of the preconditioners rely on this.
		/// Another reason for sorting is that it's more cache friendly when matrix vector multiplication is done
		/// Do not expose publicly this property, users of this class should not rely that the columns are sorted
		/// This is of length number of non-zero entries
		std::unique_ptr<int[]> positions;
		/// i-th element is index in positions and values where the i-th row starts
		/// The last element contains the number of nonzero entries of the matrix
		/// This is of length equal to the number of rows in the matrix
		std::unique_ptr<int[]> start;
		int denseRowCount; ///< Number of rows in the matrix
		int denseColCount; ///< Number of columns in the matrix
		/// Index in start array. The first row which has nonzero element in it.
		int firstActiveStart;
		/// Fill start, positions and values array. The arrays must be allocated with the right sizes before this is called.
		const int fillArrays(const TripletMatrix& triplet) noexcept;
		/// Get the next row which has at least one element in it
		/// @param[in] currentStartIndex The current row
		/// @param[in] startLength The length of the start array (same as the number of rows)
		/// @returns The next non empty row
		const int getNextStartIndex(int currentStartIndex, int startLength) const noexcept;
		/// @brief Generic function to which will perfrom out = op(lhs, A * mult).
		/// It is allowed out to be the same pointer as lhs.
		/// @tparam FunctorType type for a functor implementing operator(real* lhs, real* rhs)
		/// @param[in] lhs Left hand side operand.
		/// @param[in] mult The vector which will multiply the matrix (to the right).
		/// @param[out] out Preallocated vector where the result will be stored.
		/// @param[in] op Functor which will execute op(lhs, A * mult) it must take in two real vectors.
		template<typename FunctorType>
		void rMultOp(const real* const lhs, const real* const mult, real* const out, const FunctorType& op) const noexcept;

		#if defined(SMM_MULTITHREADING_CPPTM)
		/// @brief Generic function to which will perfrom out = op(lhs, A * mult).
		/// It is allowed out to be the same pointer as lhs.
		/// @tparam FunctorType type for a functor implementing operator(real* lhs, real* rhs)
		/// @param[in] lhs Left hand side operand.
		/// @param[in] mult The vector which will multiply the matrix (to the right).
		/// @param[out] out Preallocated vector where the result will be stored.
		/// @param[in] op Functor which will execute op(lhs, A * mult) it must take in two real vectors.
		/// @param[in] tm The thred manager which will run the multithreaded job
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		template<typename FunctorType>
		void rMultOp(
			const real* const lhs,
			const real* const mult,
			real* const out,
			const FunctorType& op,
			CPPTM::ThreadManager& tm,
			const bool async
		) const noexcept;
		#endif
	};

	inline CSRMatrix::CSRMatrix() noexcept :
		values(nullptr),
		positions(nullptr),
		start(nullptr),
		denseRowCount(0),
		denseColCount(0),
		firstActiveStart(-1)
	{ }

	inline CSRMatrix::CSRMatrix(const TripletMatrix& triplet) noexcept :
		values(new real[triplet.getNonZeroCount()]),
		positions(new int[triplet.getNonZeroCount()]),
		start(new int[triplet.getDenseRowCount() + 1]),
		denseRowCount(triplet.getDenseRowCount()),
		denseColCount(triplet.getDenseColCount()),
		firstActiveStart(-1)
	{
		fillArrays(triplet);
	}

	inline int CSRMatrix::init(const TripletMatrix& triplet) noexcept {
		denseRowCount = triplet.getDenseRowCount();
		denseColCount = triplet.getDenseColCount();
		const int nnz = triplet.getNonZeroCount();
		values.reset(new real[nnz]);
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

	inline int CSRMatrix::getNonZeroCount() const noexcept {
		return start[getDenseRowCount()];
	}

	inline int CSRMatrix::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	inline int CSRMatrix::getDenseColCount() const noexcept {
		return denseColCount;
	}

	inline bool CSRMatrix::hasSameNonZeroPattern(const CSRMatrix& other) {
		// This function checks if the two matrices have the same non zero pattern.
		// It relies that all CSR matrices will order their elements the same way.
		// For example it will not work if the elements in positions are the same, but
		// reordered between the two matrices.

		const int denseRowCount = getDenseRowCount();
		if(denseRowCount != other.getDenseRowCount()) return false;

		if(getDenseColCount() != other.getDenseColCount()) return false;

		const int nonZeroCount = getNonZeroCount();
		if(nonZeroCount != other.getNonZeroCount()) return false;

		if(memcmp(start.get(), other.start.get(), sizeof(int) * denseRowCount) != 0) return false;
		if(memcmp(positions.get(), other.positions.get(), sizeof(int) * nonZeroCount) != 0) return false;

		return true;
	}

	inline CSRMatrix::ConstIterator CSRMatrix::begin() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), firstActiveStart, 0);
	}

	inline CSRMatrix::ConstIterator CSRMatrix::end() const noexcept {
		return ConstIterator(values.get(), positions.get(), start.get(), denseRowCount, start[denseRowCount]);
	}

	template<typename FunctorType>
	inline void CSRMatrix::rMultOp(
		const real* const lhs,
		const real* const mult,
		real* const out,
		const FunctorType& op
	) const noexcept {
		for (int row = firstActiveStart; row < denseRowCount; row = getNextStartIndex(row, denseRowCount)) {
			real dot = 0.0f;
			for (int colIdx = start[row]; colIdx < start[row + 1]; ++colIdx) {
				const int col = positions[colIdx];
				const real val = values[colIdx];
				dot = _smm_fma(val, mult[col], dot);
			}
			out[row] = op(lhs[row], dot);
		}
	}

	inline void CSRMatrix::rMult(const real* const mult, real* const res) const noexcept {
		assert(mult != res);
		auto rhsId = [](const real lhs, const  real rhs) -> real {
			return rhs;
		};
		rMultOp(res, mult, res, rhsId);
	}

	inline void CSRMatrix::rMultAdd(const real* const lhs, const real* const mult, real* const out) const noexcept {
		auto addOp = [](const real lhs, const real rhs) ->real {
			return lhs + rhs;
		};
		rMultOp(lhs, mult, out, addOp);
	}


	inline void CSRMatrix::rMultSub(const real* const lhs, const real* const mult, real* const out) const noexcept {
		auto addOp = [](const real lhs, const real rhs) ->real {
			return lhs - rhs;
		};
		rMultOp(lhs, mult, out, addOp);
	}


// ********************************************************************
// *********** MULTITHREADED VARIANTS OF MATRIX FUNCTIONS *************
// ********************************************************************
#ifdef SMM_MULTITHREADING_CPPTM
	template<typename FunctorType>
	inline void CSRMatrix::rMultOp(
		const real* const lhs,
		const real* const mult,
		real* const out,
		const FunctorType& op,
		CPPTM::ThreadManager& tm,
		const bool async
	) const noexcept {
		auto rMultOpTask = [&,this](const int blockIndex, const int numBlocks) {
			const int blockSize = (denseRowCount + numBlocks) / numBlocks;
			const int startIdx = blockSize * blockIndex;
			const int end = std::min(denseRowCount, startIdx + blockSize);
			const int start = startIdx == 0 ? firstActiveStart : getNextStartIndex(startIdx - 1, denseRowCount);
			for (int row = start; row < end; row = getNextStartIndex(row, denseRowCount)) {
				real dot = 0.0f;
				for (int colIdx = this->start[row]; colIdx < this->start[row + 1]; ++colIdx) {
					const int col = positions[colIdx];
					const real val = values[colIdx];
					dot = _smm_fma(val, mult[col], dot);
				}
				out[row] = op(lhs[row], dot);
			}
		};
		if (async) {
			tm.launchAsync(std::move(rMultOpTask));
		} else {
			tm.launchSync(rMultOpTask);
		}
	}

	inline void CSRMatrix::rMult(
		const real* const mult,
		real* const res,
		CPPTM::ThreadManager& tm,
		const bool async
	) const noexcept {
		assert(mult != res);
		auto rhsId = [](const real lhs, const  real rhs) -> real {
			return rhs;
		};
		rMultOp(res, mult, res, rhsId, tm, async);
	}

	inline void CSRMatrix::rMultAdd(
		const real* const lhs,
		const real* const mult,
		real* const out,
		CPPTM::ThreadManager& tm,
		const bool async
	) const noexcept {
		auto addOp = [](const real lhs, const real rhs) ->real {
			return lhs + rhs;
		};
		rMultOp(lhs, mult, out, addOp, tm, async);
	}


	inline void CSRMatrix::rMultSub(
		const real* const lhs,
		const real* const mult,
		real* const out,
		CPPTM::ThreadManager& tm,
		const bool async
	) const noexcept {
		auto addOp = [](const real lhs, const real rhs) ->real {
			return lhs - rhs;
		};
		rMultOp(lhs, mult, out, addOp, tm, async);
	}
#endif

	inline const int CSRMatrix::getNextStartIndex(int currentStartIndex, int startLength) const noexcept {
		do {
			currentStartIndex++;
		} while (currentStartIndex < startLength && start[currentStartIndex] == start[currentStartIndex + 1]);
		return currentStartIndex;
	}

	inline void CSRMatrix::operator*=(const real scalar) {
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] *= scalar;
		}
	}

	inline void CSRMatrix::inplaceAdd(const CSRMatrix& other) {
		assert(hasSameNonZeroPattern(other) && "The two matrices have different nonzero patterns");
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] += other.values[i];
		}
	}

	inline void CSRMatrix::inplaceSubtract(const CSRMatrix& other) {
		assert(hasSameNonZeroPattern(other) && "The two matrices have different nonzero patterns");
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] -= other.values[i];
		}
	}

	inline bool CSRMatrix::updateEntry(const int row, const int col, const real newValue) {
		// Assumes that the columns are sorted in increasing order
		int rowBegin = positions[start[row]];
		int rowEnd = positions[start[row + 1]] - 1;
		while(rowBegin <= rowEnd) {
			const int mid = (rowBegin + rowEnd) / 2;
			const int currentColumn = positions[mid];
			if(currentColumn > col) {
				rowBegin = mid + 1;
			} else if(currentColumn < col) {
				rowEnd = mid - 1;
			} else {
				assert(currentColumn == col);
				values[mid] = newValue;
				return true;
			}
		}
		return false;
	}

	inline const int CSRMatrix::fillArrays(const TripletMatrix& triplet) noexcept {
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
			const int row = el.getRow();
			const int currentCount = count[row];
			const int position = start[row + 1] - currentCount;
			// Columns in each row are sorted in increasing order.
			assert(position == start[row] || positions[position - 1] < el.getCol());
			positions[position] = el.getCol();
			values[position] = el.getValue();
			count[row]--;
		}
		return 0;
	}

	template<SolverPreconditioner precond>
	inline decltype(auto) CSRMatrix::getPreconditioner() const noexcept {
		if constexpr (precond == SolverPreconditioner::NONE) {
			return IDPreconditioner();
		} else if constexpr (precond == SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL) {
			return SGSPreconditioner(*this);
		}
	}

	inline CSRMatrix::SGSPreconditioner::SGSPreconditioner(const CSRMatrix& m) noexcept :
		m(m)
	{}

	inline const int CSRMatrix::SGSPreconditioner::apply(const real* rhs, real* x) const noexcept {
		// The symmetric Gaus-Seidel comes in the form M = (D + L)D^-1(D + U)
		// We want to find x=M^{-1}rhs, note however that (D + L) is lower triangular matrix
		// D^-1(D + U) is upper trianguar matrix, thus it can be rewritten as Mx=rhs and solved in two
		// steps (D + L)y = rhs, and then (I - D^{-1}U)x=y. 

		// Assumes that the matrix has full structural rank. Thus there are no leading empty rows
		assert(m.firstActiveStart == 0);
		assert(rhs != x);
		if(m.firstActiveStart != 0) {
			return 1;
		}

		// 1. Forward substitution: (D + L)x=rhs
		for(int row = 0; row < m.getDenseRowCount(); ++row) {
			// The CSR matrix is sorted by increasing column indexes
			int indexInRow = m.start[row];
			const int nonZerosInRow = m.start[row + 1] - indexInRow;
			assert(nonZerosInRow > 0);
			if(nonZerosInRow == 0) {
				return 1;
			}
			int col = m.positions[indexInRow];
			real value = m.values[indexInRow];
			real lhs = rhs[row];
			while(col < row) {
				lhs = _smm_fma(-value, x[col], lhs);
				++indexInRow;
				col = m.positions[indexInRow];
				value = m.values[indexInRow];
			}
			assert(col == row && std::abs(value) > 1e-5);
			if(col != row || std::abs(value) < 1e-5) {
				return 1;
			}
			x[row] = lhs / value;
		}

		// 2. Backsubstitution: (I + D^{-1}U)x=x, x will be computed inplace
		for(int row = m.getDenseRowCount() - 1; row >= 0; --row) {
			int indexInRow = m.start[row + 1] - 1;
			int col = m.positions[indexInRow];
			real value = m.values[indexInRow];
			real lhs(0);
			while(col > row) {
				lhs = _smm_fma(value, x[col], lhs);
				--indexInRow;
				col = m.positions[indexInRow];
				value = m.values[indexInRow];
			}
			assert(col == row);
			x[row] = x[row] - lhs / value;
		}
		return 0;
	}

	inline CSRMatrix::ILU0Preconditioner::ILU0Preconditioner(const CSRMatrix& m) noexcept : 
		m(m),
		ilu0Val(std::make_unique<real[]>(m.getNonZeroCount()))
	{	}

	inline const int CSRMatrix::ILU0Preconditioner::validate() noexcept {
		return factorize();
	}

	inline int CSRMatrix::ILU0Preconditioner::factorize() noexcept {
		// L and U will have the same non zero pattern as the lower and upper triangular parts of m
		// The decompozition will take the form m = L * U + R, where we shall take only L and U and 
		// m - L * U will be zero for all non zero elements of m, but m - L * U might have some non zero
		// elements in places where m had zeros.

		const int rows = m.getDenseRowCount();
		const int cols = m.getDenseColCount();
		assert(rows == cols);
		memcpy(ilu0Val.get(), m.values.get(), sizeof(real) * m.getNonZeroCount());
		assert(m.firstActiveStart == 0 && "The matrix does not have full rank");
		if(m.firstActiveStart != 0) {
			return 1;
		}
		// TODO [ILU0 Reordering]: Currently we assume that the ILU(0) factorization exists, for the current matrix as it is
		// However for some matrices reordering might be needed, this is not handled for now.
		assert(m.positions[0] == 0 && "The matrix has zero on the main diagonal. Reordering is needed.");
		if(m.positions[0] == 0) {
			return 2;
		}
		// This is used in the most inner loop of the following factorization, columnIndex[i] will be the index in m.positions of the i-th
		// column in a row or -1 if the column is zero. This will be used to track the columns in the most outer loop in the factorization.
		Vector columnIndex(cols, -1);
		// TODO [Move Diagonal To End]: This can be avoided if the diagonal elements are kept in a fixed position in each row
		// For example keep the diagonal element in the end of the row.
		Vector diagonalElementsInv(rows);
		diagonalElementsInv[0] = real(1) / m.values[0];
		// The algorithm assumes that the columns in each row are sorted in increasing order
		// U will have explicit main diagonal, L will have implicit main diagonal filled with 1
		// The first row is trivial to compute, as u_{i,j} = m_{i,j} / l_{i,j}, but l_{i,j} = 1
		for(int row = 1; row < rows; ++row) {
			const int rowStart = m.start[row];
			const int rowEnd = m.start[row + 1];
			// Save the indexes for each non zero column in the row
			for(int i = rowStart; i < rowEnd; ++i) {
				const int column = m.positions[i];
				columnIndex[column] = i;
			}
			int kPos = rowStart;
			int k = m.positions[kPos];
			for(; k < row; k = m.positions[++kPos]) {
				const real alphaIK = ilu0Val[kPos] * diagonalElementsInv[k];
				ilu0Val[kPos] = alphaIK;
				for(int colPos = m.start[k + 1] - 1, col = m.positions[colPos]; col > 0; col = m.positions[--colPos]) {
					const real betaKJ = ilu0Val[colPos];
					if(columnIndex[col] != -1) {
						ilu0Val[columnIndex[col]] -= alphaIK * betaKJ;
					}
				}
			}
			assert(k == row && ilu0Val[kPos] > 1e-6f && "Zero in pivot position!");
			if(k == row && ilu0Val[kPos] > 1e-6f) {
				return 2;
			}
			diagonalElementsInv[k] = real(1) / ilu0Val[kPos];
			// Clear the column indexes and prepare them for the next iterations
			for(int i = rowStart; i < rowEnd; ++i) {
				const int column = m.positions[i];
				columnIndex[column] = -1;
			}
		}

		return 0;
	}

	inline void saveDenseText(const char* filepath, const CSRMatrix& m) {
		std::ofstream file(filepath);
		if (!file.is_open()) {
			return;
		}
		file << std::fixed << std::setprecision(6);
		const auto writeZeroes = [&file](int numZeroes) -> void {
			for (int i = 0; i < numZeroes; ++i) {
				file << "0";
				if (i < numZeroes - 1) {
					file << ",";
				}
			}
		};

		const auto writeEmptyRows = [&file, cols=m.getDenseColCount(), &writeZeroes](int numRows) -> void {
			for (int i = 0; i < numRows; ++i) {
				file << "{"; writeZeroes(cols); file << "}";
				if (i < numRows - 1) {
					file << ",\n";
				}
			}
		};

		int lastRow = -1, lastCol = -1;
		CSRMatrix::ConstIterator it = m.begin();
		file << m.getDenseRowCount() << " " << m.getDenseColCount() << "\n";
		file << "{\n";
		while (it != m.end()) {
			const int emptyRows = it->getRow() - lastRow - 1;
			lastCol = -1;
			writeEmptyRows(emptyRows);
			if (emptyRows && it->getRow() && lastRow != m.getDenseRowCount() - 1) {
				file << ",\n";
			}
			lastRow = it->getRow();
			file << "{";
			int emptyCols = 0;
			while (it != m.end() && it->getRow() == lastRow) {
				emptyCols = it->getCol() - lastCol - 1;
				lastCol = it->getCol();
				writeZeroes(emptyCols);
				if (lastCol > 0 && emptyCols) {
					file << ",";
				}
				file << it->getValue();
				if (it->getCol() != m.getDenseColCount() - 1) {
					file << ",";
				}
				++it;
			}
			emptyCols = m.getDenseColCount() - lastCol - 1;
			writeZeroes(emptyCols);
			file << "}";
			if (lastRow < m.getDenseRowCount() - 1) {
				file << ",";
			}
			file << "\n";
		}
		const int emptyRows = m.getDenseRowCount() - lastRow - 1;
		writeEmptyRows(emptyRows);
		file << "}";
	}

	/// @brief Function to convert matrix from compressed sparse row format to dense row major matrix
	/// Out must be allocated and filled with zero before being passed to the function
	/// Out will be contain linearized dense version of the matrix, first CSRMatrix::getDenseRowCount elements
	/// will be the first row of the dense matrix, next CSRMatrix::getDenseRowCount will be the second row and so on.
	/// @param[in] compressed Matrix in Compressed Sparse Row format
	/// @param[out] out Preallocated (and filled with zero) space where the dense matrix will be added
	template<typename CompressedMatrixFormat>
	inline void toLinearDenseRowMajor(const CompressedMatrixFormat& compressed, real* out) noexcept {
		const int64_t colCount = compressed.getDenseColCount();
		for (const auto& el : compressed) {
			const int64_t index = el.getRow() * colCount + el.getCol();
			out[index] = el.getValue();
		}
	}

	enum class SolverStatus {
		SUCCESS = 0,
		DIVERGED,
		MAX_ITERATIONS_REACHED
	};

	/// @brief solve a.x=b using BiConjugate Gradient method, where matrix a is symmetric
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too 
	/// @return SolverStatus the status the solved system
	inline SolverStatus BiCGSymmetric(
		const CSRMatrix& a,
		real* b,
		real* x,
		int maxIterations,
		real eps
	) {

		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		Vector r(a.getDenseRowCount());
		a.rMultSub(b, x, r);

		Vector p(a.getDenseRowCount());
		for (int i = 0; i < p.getSize(); ++i) {
			p[i] = r[i];
		}

		Vector ap(a.getDenseRowCount());

		real rSquare = r * r;
		int iterations = 0;
		real infNorm = real(0);
		do {
			a.rMult(p, ap);
			const real denom = ap* p;
#ifdef SMM_DEBUG_PRINT
			std::cout << "i: " << iterations << std::endl;
			std::cout << "r^2: " << rSquare << std::endl;
			std::cout << "Ap.p: " << denom << std::endl;
#endif
			// Numerical instability will cause devision by zero (or something close to). The method must be restarted
			// For positive definite matrices if denom becomes 0 this is a lucky breakdown so we should not exit with error
			// but continue iterating. However we cannot know in advance if the matrix is positive definite, thus a heuristic is used.
			// If a system with positive definite matrix is solved, near the lucky breakdown the residual must be small, so it's length
			// squared will be small too, so for rSquare and small denom we continue, if the rSquare is large we are most likely dealing
			// with indefinite matrix and this is serious breakdown so we exit the procedure with error message.
			if (eps > std::abs(denom) && rSquare > 1) {
				return SolverStatus::DIVERGED;
			}
			const real alpha = rSquare / denom;
			infNorm = real(0);
			for(int i = 0; i < a.getDenseRowCount(); ++i) {
				x[i] += alpha * p[i];
				r[i] -= alpha * ap[i];
				infNorm = std::max(std::abs(r[i]), infNorm);
			}
			// Dot product r * r can be zero (or close to zero) only if r has length close to zero.
			// But if the residual is close to zero, this means that we have found a solution
			const real newRSquare = r * r;
#ifdef SMM_DEBUG_PRINT
			std::cout << "alpha: " << alpha << std::endl;
			std::cout << "new r^2: " << newRSquare << std::endl;
#endif
			// If rSquare is small it's expected next iteration residual to be small too
			// Thus deleting large number by a small is highly unlikely here
			// If rSquare is small and newRSquare is large, we have critical brakedown, which might happen with BiCG method
			if(newRSquare > 1 && rSquare < eps) {
				return SolverStatus::DIVERGED;
			}
			const real beta = newRSquare / rSquare;
#ifdef SMM_DEBUG_PRINT
			std::cout << "beta: " << beta << std::endl;
			std::cout << "==================================================" << std::endl;
#endif
			for(int i = 0; i < a.getDenseRowCount(); ++i) {
				p[i] = r[i] + beta * p[i];
			}
			rSquare = newRSquare;
			iterations++;
		} while ((infNorm > eps || rSquare > eps * eps) && iterations < maxIterations);

		if (iterations > maxIterations) {
			return SolverStatus::MAX_ITERATIONS_REACHED;
		}
		return SolverStatus::SUCCESS;
	}

	/// @brief solve a.x=b using BiConjugate Gradient Squared method
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too 
	/// @return SolverStatus the status the solved system
	inline SolverStatus BiCGSquared(const CSRMatrix& a, real* b, real* x, int maxIterations, real eps) {
		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		const int rows = a.getDenseRowCount();
		Vector r(rows), r0(rows);
		a.rMultSub(b, x, r);
		
		// Help vectors, as in Saad's book, the vectors in the polynomial reccursion are: q, p, r
		// They can be expressed only in terms of themselves, the other vectors do same some computation
		// of the use a lot of memory they can be removed.
		Vector p(rows), u(rows), q(rows), alphaUQ(rows), ap(rows);
		for(int i = 0; i < rows; ++i) {
			p[i] = r[i];
			u[i] = r[i];
			r0[i] = r[i];
		}

		real rr0 = r * r0;
		real infNorm = real(0);
		int iterations = 0;
		do {
			a.rMult(p, ap);
			const real denom = ap * r0;
			// Must investigate if denom < eps is critical breakdown
			const real alpha = rr0 / denom;
			for(int i = 0; i < rows; ++i) {
				q[i] = _smm_fma(-alpha, ap[i], u[i]);
				alphaUQ[i] = alpha * (u[i] + q[i]);
				x[i] = x[i] + alphaUQ[i];
			}
			a.rMultSub(r, alphaUQ, r);
			const real newRR0 = r * r0;
			// Must investigate if rr0 < eps is critical breakdown
			const real beta = newRR0 / rr0;
			
			infNorm = real(0);
			for(int i = 0; i < rows; ++i) {
				u[i] = _smm_fma(beta, q[i], r[i]);
				p[i] = _smm_fma(beta, _smm_fma(beta, p[i], q[i]), u[i]);
				infNorm = std::max(std::abs(r[i]), infNorm);
			}
			rr0 = newRR0;
			iterations++;
		} while (infNorm > eps && iterations < maxIterations);

		if (iterations > maxIterations) {
			return SolverStatus::MAX_ITERATIONS_REACHED;
		}
		return SolverStatus::SUCCESS;
	}

	/// @brief solve a.x=b using BiConjugate Gradient Stabilized method
	/// @tparam Preconditioner Type of the preconditioner which will be applied to this matrix
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too 
	/// @param[in] maxIterations Iterations threshold for the method.
	/// If convergence was not reached for less than maxIterations the method will exit.
	/// If maxIterations is -1 the method will do all possible iterations (the same as the number of rows in the matrix)
	/// @param[in] eps Required size of the L2 norm of the residual
	/// @param[in] precond Preconditioner class which will be applied to the current system
	/// @return SolverStatus the status the solved system
	template<typename Preconditioner>
	inline SolverStatus BiCGStab(
		const CSRMatrix& a,
		real* b,
		real* x,
		int maxIterations,
		real eps,
		const Preconditioner& preconditioner
	) {
		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		const int rows = a.getDenseRowCount();
		// This vector is allocated only if there is some preconditioner different than the identity
		// It is used to store intermediate data needed by the preconditioner
		Vector precondScratchpad;
		constexpr bool precondition = !std::is_same<Preconditioner, decltype(a.getPreconditioner<SolverPreconditioner::NONE>())>::value;
		if constexpr(precondition) {
			precondScratchpad.init(rows);
		}

		Vector r(rows), r0(rows), p(rows), ap(rows), s(rows), as(rows);
		a.rMultSub(b, x, r);

		if constexpr(precondition) {
			preconditioner.apply(r, precondScratchpad);
		}
		
		for(int i = 0; i < rows; ++i) {
			if constexpr(precondition) {
				r[i] = precondScratchpad[i];
			}
			r0[i] = r[i];
			p[i] = r[i];
		}

		real resL2Norm = real(0);
		int iterations = 0;
		real rr0 = r * r0;
		do {
			if constexpr(precondition) {
				a.rMult(p, precondScratchpad);
				const int err = preconditioner.apply(precondScratchpad, ap);
				// Assert that the preconditioner has completed successfully.
				// TODO: Handle the case when it does not
				assert(err == 0);
			} else {
				a.rMult(p, ap);
			}

			real denom = ap * r0;
			const real alpha = rr0 / denom;
			for(int i = 0; i < rows; ++i) {
				s[i] = _smm_fma(-alpha, ap[i], r[i]);
			}

			if constexpr(precondition) {
				a.rMult(s, precondScratchpad);
				const int err = preconditioner.apply(precondScratchpad, as);
				// Assert that the preconditioner has completed successfully.
				// TODO: Handle the case when it does not
				assert(err == 0);
			} else {
				a.rMult(s, as);
			}

			denom = as * as;
			// TODO: add proper check for division by zero
			const real omega = (as * s) / denom;
			resL2Norm = real(0);
			for(int i = 0; i < rows; ++i) {
				x[i] = _smm_fma(alpha, p[i], _smm_fma(omega, s[i], x[i])); 
				r[i] = _smm_fma(-omega, as[i], s[i]); 
				resL2Norm += r[i] * r[i];
			}
			resL2Norm = std::sqrt(resL2Norm);
			const real newRR0 = r * r0;
			// TODO: add proper check for division by zero
			const real beta = (newRR0 * alpha) / (rr0 * omega);
			for(int i = 0; i < rows; ++i) {
				p[i] = _smm_fma(beta, (_smm_fma(-omega, ap[i], p[i])), r[i]);
			}
			rr0 = newRR0;
			iterations++;
		} while(resL2Norm > eps && iterations < maxIterations);

		if (iterations > maxIterations) {
			return SolverStatus::MAX_ITERATIONS_REACHED;
		}
		return SolverStatus::SUCCESS;
	}

	/// @brief solve a.x=b using BiConjugate Gradient Stabilized method
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too
	/// @param[in] maxIterations Iterations threshold for the method.
	/// If convergence was not reached for less than maxIterations the method will exit.
	/// If maxIterations is -1 the method will do all possible iterations (the same as the number of rows in the matrix)
	/// @param[in] eps Required size of the L2 norm of the residual
	/// @return SolverStatus the status the solved system
	inline SolverStatus BiCGStab(
		const CSRMatrix& a,
		real* b,
		real* x,
		int maxIterations,
		real eps
	) {
		return BiCGStab(a, b, x, maxIterations, eps, a.getPreconditioner<SolverPreconditioner::NONE>());
	}

	///@brief Solve a.x=b using Cojugate Gradient method
	/// Matrix a should be symmetric positive definite matrix. 
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in,out] x Initial condition, the result will be written here too
	/// @param[in] maxIterations Iterations threshold for the method.
	/// If convergence was not reached for less than maxIterations the method will exit.
	/// If maxIterations is -1 the method will do all possible iterations (the same as the number of rows in the matrix)
	/// @param[in] eps Required size of the L2 norm of the residual
	/// @return SolverStatus the status the solved system
	inline SolverStatus ConjugateGradient(
		const CSRMatrix& a,
		real* b,
		real* x,
		int maxIterations,
		real eps
	) {
		const int rows = a.getDenseRowCount();
		Vector r(rows);
		a.rMultSub(b, x, r);

		Vector p(rows), Ap(rows);
		for(int i = 0; i < rows; ++i) {
			p[i] = r[i];
		}
		const real epsSq = eps * eps;
		real residualNormSquared = r * r;
		if(maxIterations == -1) {
			maxIterations = rows;
		}
		for(int i = 0; i < maxIterations; ++i) {
			a.rMult(p, Ap);
			const real pAp = Ap * p;
			// If the denominator is 0 we have a lucky breakdown. The residual at the previous step must be 0.
			if(eps > pAp) {
				return SolverStatus::SUCCESS;
			}
			// alpha = (r_i, r_i) / (Ap, p)
			const real alpha = residualNormSquared / pAp;
			// x = x + alpha * p
			// r = r - alpha * Ap
			for(int j = 0; j < rows; ++j) {
				x[j] = _smm_fma(alpha, p[j], x[j]);
				r[j] = _smm_fma(-alpha, Ap[j], r[j]);
			}
			const real newResidualNormSquared = r * r;
			// beta = (r_{i+1}, r_(i+1)) / (r_i, r_i)
			const real beta = newResidualNormSquared / residualNormSquared;
			// p = r + beta * p
			for(int j = 0; j < rows; ++j) {
				p[j] = _smm_fma(beta, p[j], r[j]);
			}
			residualNormSquared = newResidualNormSquared;
			if(epsSq > residualNormSquared) {
				return SolverStatus::SUCCESS;
			}
		}
		return SolverStatus::MAX_ITERATIONS_REACHED;
	}
	
	enum class MatrixLoadStatus {
		// Generic success code
		SUCCESS = 0,
		// Generic failure codes
		FAILED_TO_OPEN_FILE,
		FAILED_TO_OPEN_FILE_UNKNOWN_FORMAT,
		FAILED_TO_PARSE_FILE,

		// Error codes for MMX files
		PARSE_ERROR_MMX_FILE_MISSING_BANNER,
		PARSE_ERROR_MMX_FILE_UNSUPPORTED_TYPE,
		PARSE_ERROR_MMX_FILE_UNSUPPORTED_FORMAT,
		PARSE_ERROR_MMX_FILE_UNSUPPORTED_EL_TYPE,
		PARSE_ERROR_MMX_FILE_UNSUPPORTED_STRUCTURE

	};

	/// @brief Load matrix in matrix market format
	/// The format has comments starting with %
	/// First actual rows is the number of rows number of cols and number of non zero elements
	/// Next number of non zero elements represent element in coordinate form (row, col, value)
	/// @param filename Path to the file with the matrix
	/// @param out Matrix in triplet (coordinate) for containing the data from the file
	/// @return MatrixLoadStatus Error code for the function
	inline MatrixLoadStatus loadMatrixMarketMatrix(const char* filepath, TripletMatrix& out) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			return MatrixLoadStatus::FAILED_TO_OPEN_FILE;
		}

		auto tolower = [](std::string& str) {
			for (char& c : str) {
				c = std::tolower(static_cast<unsigned char>(c));
			}
		};

		// Read banner
		std::string banner, matrix, format, type, structure;
		file >> banner;
		if (banner != "%%MatrixMarket") {
			return MatrixLoadStatus::PARSE_ERROR_MMX_FILE_MISSING_BANNER;
		}

		file >> matrix;
		tolower(matrix);
		if (matrix != "matrix") {
			return MatrixLoadStatus::PARSE_ERROR_MMX_FILE_UNSUPPORTED_TYPE;
		}

		file >> format;
		tolower(format);
		if (format != "coordinate") {
			return MatrixLoadStatus::PARSE_ERROR_MMX_FILE_UNSUPPORTED_FORMAT;
		}

		file >> type;
		tolower(type);
		if (type != "real" && type != "integer") {
			return MatrixLoadStatus::PARSE_ERROR_MMX_FILE_UNSUPPORTED_EL_TYPE;
		}

		file >> structure;
		tolower(structure);
		if (structure != "symmetric") {
			return MatrixLoadStatus::PARSE_ERROR_MMX_FILE_UNSUPPORTED_STRUCTURE;
		}
		
		// Skip comment section
		while (file.peek() == '%' || std::isspace(file.peek())) {
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		// Read matrix size
		int rows, cols, nnz;
		file >> rows >> cols >> nnz;
		if (file.fail()) {
			return MatrixLoadStatus::FAILED_TO_PARSE_FILE;
		}
		out.init(rows, cols, nnz);
		// Read matrix
		int filerow = 0;
		while (!file.eof()) {
			int row, col;
			real value;
			file >> row >> col >> value;
			if (file.fail()) {
				return MatrixLoadStatus::FAILED_TO_PARSE_FILE;
			}
			// In file format indexes start from 0
			row -= 1; col -= 1;
			// Current implementation supports only symmetric matrices
			out.addEntry(row, col, value);
			if (row != col) {
				out.addEntry(col, row, value);
			}
			// Hacky way to ignore empty lines. This will ignore anything that starts with an empty space/tab/newline
			while (std::isspace(file.peek())) {
				file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
			}
			filerow++;
		}
		return MatrixLoadStatus::SUCCESS;
	}

	inline MatrixLoadStatus loadSMMDTMatrix(const char* filepath, TripletMatrix& out) {
		std::ifstream file(filepath);
		if (!file.is_open()) {
			return MatrixLoadStatus::FAILED_TO_OPEN_FILE;
		}

		int rows, cols;
		file >> rows >> cols;
		if (file.fail()) {
			return MatrixLoadStatus::FAILED_TO_PARSE_FILE;
		}

		out.init(rows, cols, 0);
		file.ignore(std::numeric_limits<std::streamsize>::max(), '{');
		for (int i = 0; i < rows; ++i) {
			file.ignore(std::numeric_limits<std::streamsize>::max(), '{');
			for (int j = 0; j < cols; ++j) {
				real val;
				file >> val;
				if (file.fail()) {
					return MatrixLoadStatus::FAILED_TO_PARSE_FILE;
				}
				if (val != 0) {
					out.addEntry(i, j, val);
				}
				file.ignore(1, ',');
			}
			file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		}
		file.ignore(std::numeric_limits<std::streamsize>::max(), '\n');
		if (file.fail()) {
			return MatrixLoadStatus::FAILED_TO_PARSE_FILE;
		}
		return MatrixLoadStatus::SUCCESS;
	}

	inline MatrixLoadStatus loadMatrix(const char* filepath, TripletMatrix& out) {
		const char* fileExtension = strrchr(filepath, '.') + 1;
		if (strcmp(fileExtension, "mtx") == 0) {
			return loadMatrixMarketMatrix(filepath, out);
		} else if (strcmp(fileExtension, "smmdt") == 0) {
			return loadSMMDTMatrix(filepath, out);
		} else {
			return MatrixLoadStatus::FAILED_TO_OPEN_FILE_UNKNOWN_FORMAT;
		}
	}
}
