#pragma once
#include <vector>
#include <map>
#include <unordered_map>
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

#if defined(SMM_MULTITHREADING)
	#include <tbb/blocked_range.h>
	#include <tbb/parallel_reduce.h>
	#include <tbb/parallel_for.h>
#endif

#define SMM_MAJOR_VERSION 0
#define SMM_MINOR_VERSION 2
#define SMM_PATCH_VERSION 0

namespace SMM {
	namespace {
		/// Perform a * x + b, based on compilerd defines this might use actual fused multiply add call
		template<typename T>
		const inline T _smm_fma(T a, T x, T b) {
			#ifdef SMM_WITH_STD_FMA
				return std::fma(a, x, b);
			#else
				return a * x + b;
			#endif
		}
	}

	template<typename T>
	using do_not_deduce = std::common_type_t<T>;

	template<typename T>
	class Vector {
	public:

		using Iterator = T*;
		using ConstIterator = const T*;

		/// Default construct a vector. Sets the size to 0.
		Vector() noexcept;

		/// Construct a vector with a given size. The values will be junk, the user must init them
		/// @param[in] size The size of the vector.
		explicit Vector(const int size) noexcept;

		/// Construct a vector of given size and set all values to the specified value
		/// @param[in] size The size of the vector
		/// @param[in] val The the value which all elements of the vectori will take
		Vector(const int size, const do_not_deduce<T> val) noexcept;

		/// Construct a vector from initializer list. The vector will have size as the initializer
		/// and all elements will be copied
		/// @param[in] l The initializer list
		Vector(const std::initializer_list<T>& l) noexcept;

		/// Move construct from another vector
		Vector(Vector<T>&& other) noexcept;

		/// Move assignment from another vector
		Vector<T>& operator=(Vector<T>&& other) noexcept;

		Vector(const Vector<T>&) = delete;
		Vector<T>& operator=(const Vector<T>&) = delete;

		~Vector();

		/// Initialize the vector. It is safe to call init two (or more times). If the size of the
		/// vector is larger of than the new size there will be no allocation or deallocation of memory,
		/// only the size field will be changed. If the size of the vector is less than the new size
		/// the data will be resized accordingly.
		/// @param[in] size The number of elements the vector should have
		void init(const int size);

		/// Initialize the vector and set all elements with the passed value. It is safe to call init two (or more times).
		/// If the size of the vector is larger of than the new size there will be no allocation or deallocation of memory,
		/// only the size field will be changed. If the size of the vector is less than the new size
		/// the data will be resized accordingly.
		/// @param[in] size The number of elements the vector should have
		/// @param[in] val The value which all elements will take
		void init(const int size, const T val);

		/// Delete the data allocated by the vector
		void deinit() noexcept;

		/// @returns The number of elements in the vector
		int getSize() const;

		/// Cast the vector to a pointer of the underlying type
		operator T*();

		/// Returns a const reference to the element at specified location pos. No bounds checking is performed.
		/// @param[in] index Position of the element to return
		/// @returns Const reference to the requested element.
		const T& operator[](const int index) const;

		/// Returns a reference to the element at specified location pos. No bounds checking is performed.
		/// @param[in] index Position of the element to return
		/// @returns Reference to the requested element.
		T& operator[](const int index);

		/// Inplace addition of two vectors. The two vectors must be of the same size (no checks are performed however)
		/// @param[in] other The vector which will be added to the current one
		/// @returns Reference to this which will contain this + other
		Vector<T>& operator+=(const Vector<T>& other);

		/// Inplace subtraction of two vectors. The two vectors must be of the same size (no checks are performed however)
		/// @param[in] other The vector which will be subtracted from the current one
		/// @returns Reference to this which will contain this - other
		Vector<T>& operator-=(const Vector<T>& other);

		/// Compute the L2 (Euclidian) norm of the vector
		/// @returns L2 norm of the vector
		T secondNorm() const;

		/// Compute the L2 (Euclidian) norm of the vector squared.
		/// @returns L2 norm of the vector squared.
		T secondNormSquared() const;

		/// Compute the dot product of two vectors
		/// @param[in] other The vector which will be dotted with this
		/// @returns The dot product of this and other
		const T operator*(const Vector<T>& other) const;

		/// Iterator to the beggining of the vector
		/// @returns Iterator to the beggining of the vector
		Iterator begin() noexcept;

		/// Iterator to one past the last element. Should not be dereferenced.
		/// @returns Iterator to the end of the vector.
		Iterator end() noexcept;

		/// Constant iterator to the beggining of the vector
		/// @returns Constant iterator to the beggining of the vector
		ConstIterator begin() const noexcept;

		/// Constant terator to one past the last element. Should not be dereferenced.
		/// @returns Constant iterator to the end of the vector.
		ConstIterator end() const noexcept;

		/// Constant iterator to the beggining of the vector
		/// @returns Constant iterator to the beggining of the vector
		ConstIterator cbegin() const noexcept;

		/// Constant terator to one past the last element. Should not be dereferenced.
		/// @returns Constant iterator to the end of the vector.
		ConstIterator cend() const noexcept;

		/// Fill all elements of the vector with the current value
		/// @param[in] value The to which all elements will be set
		void fill(const T value);

		void swap(Vector<T>& other) {
			std::swap(data, other.data);
			std::swap(size, other.size);
		}
	private:
		/// Allocate new memory for the vector and set all elements to val
		void initDataWithVal(const T val);
		T* data;
		int size;
	};


	template<typename T>
	inline Vector<T>::Vector() noexcept :
		data(nullptr),
		size(0)
	{ }

	template<typename T>
	inline Vector<T>::Vector(const int sizeIn) noexcept :
		data(static_cast<T*>(malloc(sizeIn * sizeof(T)))),
		size(sizeIn)
	{ }

	template<typename T>
	inline Vector<T>::Vector(const int size, const do_not_deduce<T> val) noexcept :
		size(size)
	{
		initDataWithVal(val);
	}

	template<typename T>
	inline Vector<T>::Vector(const std::initializer_list<T>& l) noexcept :
		size(l.size()),
		data(static_cast<T*>(malloc(l.size() * sizeof(T))))
	{
		std::copy(l.begin(), l.end(), data);
	}

	template<typename T>
	inline Vector<T>::Vector(Vector<T>&& other) noexcept :
		size(other.size),
		data(other.data)
	{
		other.data = nullptr;
		other.size = 0;
	}

	template<typename T>
	inline Vector<T>& Vector<T>::operator=(Vector<T>&& other) noexcept {
		size = other.size;
		free(data);
		data = other.data;
		other.data = nullptr;
		other.size = 0;
		return *this;
	}

	template<typename T>
	inline Vector<T>::~Vector() {
		deinit();
	}

	template<typename T>
	inline void Vector<T>::init(const int size) {
		if(this->size < size) {
			data = static_cast<T*>(realloc(data, size * sizeof(T)));
			assert(data != nullptr);
		}
		this->size = size;
	}

	template<typename T>
	inline void Vector<T>::init(const int size, const T val) {
		init(size);
		fill(val);
	}

	template<typename T>
	inline void Vector<T>::deinit() noexcept {
		free(data);
		data = nullptr;
		size = 0;
	}

	template<typename T>
	inline int Vector<T>::getSize() const {
		return this->size;
	}

	template<typename T>
	inline Vector<T>::operator T*() {
		return data;
	}

	template<typename T>
	inline const T& Vector<T>::operator[](const int index) const {
		assert(index < size && index >= 0);
		return data[index];
	}

	template<typename T>
	inline T& Vector<T>::operator[](const int index) {
		assert(index < size && index >= 0);
		return data[index];
	}

	template<typename T>
	inline Vector<T>& Vector<T>::operator+=(const Vector<T>& other) {
		assert(other.size == size);
		for (int i = 0; i < size; ++i) {
			data[i] += other[i];
		}
		return *this;
	}

	template<typename T>
	inline Vector<T>& Vector<T>::operator-=(const Vector<T>& other) {
		assert(other.size == size);
		for (int i = 0; i < size; ++i) {
			data[i] -= other[i];
		}
		return *this;
	}

	template<typename T>
	inline T Vector<T>::secondNorm() const {
		T sum = 0.0;
		for (int i = 0; i < size; ++i) {
			sum += data[i] * data[i];
		}
		return std::sqrt(sum);
	}

	template<typename T>
	inline T Vector<T>::secondNormSquared() const {
		T sum(0);
		for (int i = 0; i < size; ++i) {
			sum += data[i] * data[i];
		}
		return sum;
	}

	template<typename T>
	inline const T Vector<T>::operator*(const Vector<T>& other) const {
		assert(other.size == size);
#ifdef SMM_MULTITHREADING
		const int dotProductGrainSize = 8192;
		return tbb::parallel_deterministic_reduce(
				tbb::blocked_range<int>(0, size, dotProductGrainSize),
				0.0f,
				[&](const tbb::blocked_range<int>& range, T current) {
					for(int j = range.begin(); j < range.end(); ++j) {
						current += data[j] * other[j];
					}
					return current;
				},
				std::plus<T>()
			);
#else
		T dot(0);
		for (int i = 0; i < size; ++i) {
			dot += other[i] * data[i];
		}
		return dot;
#endif
	}

	template<typename T>
	inline typename Vector<T>::Iterator Vector<T>::begin() noexcept {
		return data;
	}

	template<typename T>
	inline typename Vector<T>::Iterator Vector<T>::end() noexcept {
		return data + size;
	}

	template<typename T>
	inline typename Vector<T>::ConstIterator Vector<T>::begin() const noexcept {
		return data;
	}

	template<typename T>
	inline typename Vector<T>::ConstIterator Vector<T>::end() const noexcept {
		return data + size;
	}

	template<typename T>
	inline typename Vector<T>::ConstIterator Vector<T>::cbegin() const noexcept {
		return data;
	}

	template<typename T>
	inline typename Vector<T>::ConstIterator Vector<T>::cend() const noexcept {
		return data + size;
	}

	template<typename T>
	inline void Vector<T>::fill(const T value) {
		if(value == T(0)) {
			memset(data, 0, sizeof(T) * size);
		} else {
			std::fill_n(data, size, value);
		}
	}

	template<typename T>
	inline void Vector<T>::initDataWithVal(const T val) {
		if (val == T(0)) {
			data = static_cast<T*>(calloc(size, sizeof(T)));
		} else {
			const int64_t byteSize = int64_t(size) * sizeof(T);
			data = static_cast<T*>(malloc(byteSize));
			if (!data) return;
			for (int i = 0; i < this->size; ++i) {
				data[i] = val;
			}
		}
	}

	/// Class to represent element of a triplet matrix.
	/// @tparam Container The underlying type of the triplet matrix (The container which holds all triplets)
	/// @tparam T The type of the value of each triplet element
	template<typename Container, typename T>
	class TripletEl {

	template<typename,typename> friend class TripletMatrixConstIterator;

	public:
		TripletEl(const TripletEl&) noexcept = default;
		TripletEl& operator=(const TripletEl& other) = default;

		TripletEl(TripletEl&&) noexcept = default;
		TripletEl& operator=(TripletEl&& other) = default;

		bool operator==(const TripletEl& other) const {
			return it == other.it;
		}

		int getRow() const noexcept {
			return it->first >> 32;
		}

		int getCol() const noexcept {
			return it->first & 0xFFFFFFFF;
		}

		const T getValue() const noexcept {
			return it->second;
		}

		friend void swap(TripletEl& a, TripletEl& b) noexcept;
	private:
		TripletEl(typename Container::const_iterator it) : it(it) {

		}

		typename Container::const_iterator it;
	};

	template<typename Container, typename T>
	inline void swap(TripletEl<Container, T>& a, TripletEl<Container, T>& b) noexcept {
		using std::swap;
		swap(a.it, b.it);
	}

	template<typename Container, typename T>
	class TripletMatrixConstIterator {
	
	template<typename, typename> friend class _TripletMatrixCommon;

	public:
		using iterator_category = std::forward_iterator_tag;
		using value_type = TripletEl<Container, T>;
		using pointer = const TripletEl<Container, T>*;
		using reference = const TripletEl<Container, T>&;

		TripletMatrixConstIterator& operator=(const TripletMatrixConstIterator&) = default;

		bool operator==(const TripletMatrixConstIterator& other) const noexcept {
			return currentEl == other.currentEl;
		}

		bool operator!=(const TripletMatrixConstIterator& other) const noexcept {
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

		template<typename> friend void swap(TripletMatrixConstIterator& a, TripletMatrixConstIterator& b) noexcept;
	private:
		TripletMatrixConstIterator(typename Container::const_iterator it) : currentEl(it) {}
		TripletEl<Container, T> currentEl;
	};

	template<typename Container, typename T>
	inline void swap(TripletMatrixConstIterator<Container, T>& a, TripletMatrixConstIterator<Container, T>& b) noexcept {
		swap(a.currentEl, b.currentEl);
	}

	/// @brief Class to hold sparse matrix into triplet (coordinate) format.
	/// Triplet format represents the matrix entries as list of triplets (row, col, value)
	/// It is allowed repetition of elements, i.e. row and col can be the same for two
	/// separate entries, when this happens elements are being summed. Repeating elements does not
	/// increase the count of non zero elements. This class is supposed to be used as intermediate class to add
	/// entries dynamically. After all data is gathered it should be converted to CSRMatrix which provides
	/// various arithmetic functions.
	template<typename Container, typename T>
	class _TripletMatrixCommon {
	public:
		using value_type = T;
		using ConstIterator = TripletMatrixConstIterator<Container, T>;
		_TripletMatrixCommon();
		/// @brief Initialize triplet matrix with given number of rows and columns
		/// The number of rows and columns does not have any affect the space allocated by the matrix
		/// @param[in] rowCount Number of rows which the dense form of the matrix is supposed to have
		/// @param[in] colCount Number of columns which the dense form of the matrix is supposed to have
		_TripletMatrixCommon(int rowCount, int colCount) noexcept;
		/// @brief Initialize triplet matrix with given number of rows and columns and allocate space for the elements of the matrix
		/// Note that this constructor only allocates space but does not initialize the elements, nor it changes the number of non zero elements,
		/// thus the number of non zero elements will be 0 after the constructor is called.
		/// The number of rows and columns does not have any affect the space allocated by the matrix
		/// @param[in] rowCount Number of rows which the dense form of the matrix is supposed to have
		/// @param[in] colCount Number of columns which the dense form of the matrix is supposed to have
		/// @param[in] numTriplets How many elements to allocate space for.
		_TripletMatrixCommon(int rowCount, int colCount, int numTriplets) noexcept;
		~_TripletMatrixCommon() = default;
		_TripletMatrixCommon(_TripletMatrixCommon&&) = default;
		_TripletMatrixCommon& operator=(_TripletMatrixCommon&&) = default;
		_TripletMatrixCommon(const _TripletMatrixCommon&) = delete;
		_TripletMatrixCommon& operator=(const _TripletMatrixCommon&) = delete;
		/// Multiply each value in the matrix by the scalar inplace
		_TripletMatrixCommon<Container, T>& operator*=(const T scalar) noexcept;
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
		void addEntry(int row, int col, T value);
		/// If a non zero entry exists at position (row, col) set its value
		/// Does nothing if there is no nonzero entry at (row, col)
		/// @param[in] row The row of the entry which is going to be updated
		/// @param[in] col The column of the entry which is going to be updated
		/// @param[in] newValue The new value of the entry at (row, col)
		/// @returns True if the there is an entry at (row, col) and the update is successful
		bool updateEntry(const int row, const int col, const T newValue);
		/// Retrieve the value of the element at position (row, col)
		/// IMPORTANT: The getting element is NOT constant operation. Directly getting elements
		/// must be avoided.
		/// @param[in] row The row of the element
		/// @param[in] col The column of the element
		/// @returns The value of the element at position (row, col). Note 0 is possible result, but
		/// it does not mean that the element at (row, col) is imlicit (not in the sparse structure)
		T getValue(const int row, const int col) const;
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
		Container data;
		int denseRowCount; ///< Number of rows in the matrix  
		int denseColCount; ///< Number of columns in the matrix
	};

	template<typename Container, typename T>
	inline _TripletMatrixCommon<Container, T>::_TripletMatrixCommon() :
		denseRowCount(0),
		denseColCount(0)
	{ }

	template<typename Container, typename T>
	inline _TripletMatrixCommon<Container, T>::_TripletMatrixCommon(int denseRowCount, int denseColCount) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{ }

	template<typename Container, typename T>
	inline _TripletMatrixCommon<Container, T>::_TripletMatrixCommon(
		int denseRowCount,
		int denseColCount,
		[[maybe_unused]]int numTriplets
	) noexcept :
		denseRowCount(denseRowCount),
		denseColCount(denseColCount)
	{ }

	template<typename Container, typename T>
	inline void _TripletMatrixCommon<Container, T>::init(
		const int denseRowCount,
		const int denseColCount,
		[[maybe_unused]]const int numTriplets
	) {
		assert(getNonZeroCount() == 0 && getDenseRowCount() == 0 && getDenseColCount() == 0);
		this->denseRowCount = denseRowCount;
		this->denseColCount = denseColCount;
	}

	template<typename Container, typename T>
	inline void _TripletMatrixCommon<Container, T>::deinit() {
		denseRowCount = 0;
		denseColCount = 0;
		data.clear();
	}

	template<typename Container, typename T>
	inline void _TripletMatrixCommon<Container, T>::addEntry(int row, int col, T value) {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && col < denseColCount);
		const uint64_t key = (uint64_t(row) << 32) | uint64_t(col);
		auto it = data.find(key);
		if (it == data.end()) {
			data[key] = value;
		} else {
			it->second += value;
		}
	}

	template<typename Container, typename T>
	inline bool _TripletMatrixCommon<Container, T>::updateEntry(const int row, const int col, const T newValue) {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && col < denseColCount);
		const uint64_t key = (uint64_t(row) << 32) | uint64_t(col);
		auto it = data.find(key);
		if(it != data.end()) {
			it->second = newValue;
			return true;
		}
		return false;
	}

	template<typename Container, typename T>
	inline T _TripletMatrixCommon<Container, T>::getValue(const int row, const int col) const {
		static_assert(2 * sizeof(int) == sizeof(uint64_t), "Expected 32 bit integers");
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && col < denseColCount);
		const uint64_t key = (uint64_t(row) << 32) | uint64_t(col);
		auto it = data.find(key);
		if(it == data.end()) {
			return T(0);
		}
		return it->second;
	}

	template<typename Container, typename T>
	inline typename _TripletMatrixCommon<Container, T>::ConstIterator _TripletMatrixCommon<Container, T>::begin() const noexcept {
		return ConstIterator(data.cbegin());
	}

	template<typename Container, typename T>
	inline typename _TripletMatrixCommon<Container, T>::ConstIterator _TripletMatrixCommon<Container, T>::end() const noexcept {
		return ConstIterator(data.cend());
	}

	template<typename Container, typename T>
	inline int _TripletMatrixCommon<Container, T>::getNonZeroCount() const noexcept {
		return data.size();
	}

	template<typename Container, typename T>
	inline int _TripletMatrixCommon<Container, T>::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	template<typename Container, typename T>
	inline int _TripletMatrixCommon<Container, T>::getDenseColCount() const noexcept {
		return denseColCount;
	}

	template<typename Container, typename T>
	inline _TripletMatrixCommon<Container, T>& _TripletMatrixCommon<Container, T>::operator*=(const T scalar) noexcept {
		for(auto& kv : data) {
			kv.second *= scalar;
		}
		return *this;
	}

	template<typename T>
	using TripletMatrix = _TripletMatrixCommon<std::map<uint64_t, T>, T>;

	template<typename T>
	using UnorderedTripletMatrix = _TripletMatrixCommon<std::unordered_map<uint64_t, T>, T>;

	template<typename T>
	struct is_ptr_to_const : std::conjunction<
	    std::is_pointer<T>,
	    std::is_const<std::remove_pointer_t<T>>
	> {};

	template<typename T>
	static inline constexpr bool is_ptr_to_const_t = is_ptr_to_const<T>::value;

	template<typename TPtr, typename T = std::enable_if_t<std::is_pointer_v<TPtr>, std::remove_pointer_t<TPtr>>>
	struct make_ptr_to_const {
		typedef std::conditional_t<std::is_const_v<TPtr>, std::add_const_t<const T*>, const T*> type;
	};

	template<typename T>
	using  make_ptr_to_const_t = typename make_ptr_to_const<T>::type;


	/// @brief Base class for const forward iterator for matrix in compressed sparse row format
	template<typename MatrixPtrT>
	class _CSRIteratorBase {
	public:
		using el_value_type = typename std::remove_pointer_t<MatrixPtrT>::value_type;
		friend class _CSRIteratorBase<make_ptr_to_const_t<MatrixPtrT>>;
		class CSRElement {
		public:
			friend class _CSRIteratorBase<MatrixPtrT>;
			friend class _CSRIteratorBase<make_ptr_to_const_t<MatrixPtrT>>;
			const el_value_type getValue() const noexcept;
			int getRow() const noexcept;
			int getCol() const noexcept;
			void setValue(el_value_type value) noexcept;
			friend void swap(CSRElement& a, CSRElement& b) noexcept;
			CSRElement(
				MatrixPtrT m,
				const int currentStartIndex,
				const int currentPositionIndex
			) noexcept;

			CSRElement(const CSRElement&) = default;
			CSRElement& operator=(const CSRElement&) = default;
			bool operator==(const CSRElement&) const;

		protected:
			/// Pointer to the sparse matrix whose element this is
			MatrixPtrT m;
			/// Index into start for the element which the iterator is pointing to
			int currentStartIndex;
			/// Index into positions for the element which the iterator is pointing to
			int currentPositionIndex;
		};

		using iterator_category = std::forward_iterator_tag;
		using value_type = CSRElement;
		using pointer = std::conditional_t<is_ptr_to_const<MatrixPtrT>::value, const CSRElement*, CSRElement*>;
		using reference = std::conditional_t<is_ptr_to_const<MatrixPtrT>::value, const CSRElement&, CSRElement&>;

		_CSRIteratorBase(
			MatrixPtrT m,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;

		template<typename Other, typename = std::enable_if_t<!is_ptr_to_const_t<Other> || is_ptr_to_const_t<MatrixPtrT>>>
		_CSRIteratorBase(const _CSRIteratorBase<Other>& other) :
			currentElement(other.currentElement.m, other.currentElement.currentStartIndex, other.currentElement.currentPositionIndex)
		{}

		_CSRIteratorBase(const _CSRIteratorBase&) = default;
		_CSRIteratorBase& operator=(const _CSRIteratorBase&) = default;
		bool operator==(const _CSRIteratorBase& other) const noexcept;
		bool operator!=(const _CSRIteratorBase& other) const noexcept;
		reference operator*() const;
		pointer operator->() const;
		reference operator*();
		pointer operator->();
	protected:
		int getColumnPointer() const {
			return currentElement.currentPositionIndex;
		}
		void setColumnPointer(int cp) {
			assert(cp >= 0 && cp <= currentElement.m->getNonZeroCount());
			currentElement.currentPositionIndex = cp;
		}
		int getRow() const {
			return currentElement.currentStartIndex;
		}
		void setRow(int row) {
			assert(row >= 0 && row <= currentElement.m->getDenseRowCount());
			currentElement.currentStartIndex = row;
		}
		int getColumn() {
			return currentElement.m[currentElement.currentPositionIndex];
		}
		int getRowStartPointer(int row) {
			return currentElement.m->start[row];
		}
		int getDenseRowCount() const {
			return currentElement.m->getDenseRowCount();
		}
		CSRElement currentElement;
	};

	template<typename MatrixPtrT>
	inline _CSRIteratorBase<MatrixPtrT>::_CSRIteratorBase(
		MatrixPtrT m,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		currentElement(m, currentStartIndex, currentPositionIndex)
	{ }

	template<typename MatrixPtrT>
	inline typename _CSRIteratorBase<MatrixPtrT>::reference _CSRIteratorBase<MatrixPtrT>::operator*() const {
		return currentElement;
	}

	template<typename MatrixPtrT>
	inline typename _CSRIteratorBase<MatrixPtrT>::pointer _CSRIteratorBase<MatrixPtrT>::operator->() const {
		return &currentElement;
	}

	template<typename MatrixPtrT>
	inline typename _CSRIteratorBase<MatrixPtrT>::reference _CSRIteratorBase<MatrixPtrT>::operator*() {
		return currentElement;
	}

	template<typename MatrixPtrT>
	inline typename _CSRIteratorBase<MatrixPtrT>::pointer _CSRIteratorBase<MatrixPtrT>::operator->() {
		return &currentElement;
	}

	template<typename MatrixPtrT>
	inline bool _CSRIteratorBase<MatrixPtrT>::operator==(const _CSRIteratorBase<MatrixPtrT>& other) const noexcept {
		return currentElement == other.currentElement;
	}

	template<typename MatrixPtrT>
	inline bool _CSRIteratorBase<MatrixPtrT>::operator!=(const _CSRIteratorBase<MatrixPtrT>& other) const noexcept {
		return !(*this == other);
	}

	template<typename MatrixPtrT>
	inline _CSRIteratorBase<MatrixPtrT>::CSRElement::CSRElement(
		MatrixPtrT m,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		m(m),
		currentStartIndex(currentStartIndex),
		currentPositionIndex(currentPositionIndex)
	{
		static_assert(std::is_pointer_v<MatrixPtrT>, "Only pointer are allowed as matrix type templates.");
	}

	template<typename MatrixPtrT>
	inline const typename _CSRIteratorBase<MatrixPtrT>::el_value_type _CSRIteratorBase<MatrixPtrT>::CSRElement::getValue() const noexcept {
		return m->values[currentPositionIndex];
	}

	template<typename MatrixPtrT>
	inline void _CSRIteratorBase<MatrixPtrT>::CSRElement::setValue(const typename _CSRIteratorBase<MatrixPtrT>::el_value_type value) noexcept {
		m->values[currentPositionIndex] = value;
	}

	template<typename MatrixPtrT>
	inline int _CSRIteratorBase<MatrixPtrT>::CSRElement::getRow() const noexcept {
		return currentStartIndex;
	}

	template<typename MatrixPtrT>
	inline int _CSRIteratorBase<MatrixPtrT>::CSRElement::getCol() const noexcept {
		return m->positions[currentPositionIndex];
	}

	template<typename MatrixPtrT>
	inline void swap(
		typename _CSRIteratorBase<MatrixPtrT>::CSRElement& a,
		typename _CSRIteratorBase<MatrixPtrT>::CSRElement& b
	) noexcept {
		using std::swap;
		swap(a.m, b.m);
		swap(a.currentStartIndex, b.currentStartIndex);
		swap(a.currentPositionIndex, b.currentPositionIndex);
	}

	template<typename MatrixPtrT>
	inline bool _CSRIteratorBase<MatrixPtrT>::CSRElement::operator==(const _CSRIteratorBase<MatrixPtrT>::CSRElement& other) const {
		return m == other.m&&
			other.currentStartIndex == currentStartIndex &&
			other.currentPositionIndex == currentPositionIndex;
	}

	/// Forward iterator, which iterates over all elements of the matrix
	/// The rows are guaranteed to be iterated in an increasing order
	template<typename MatrixPtrT>
	class CSRIterator : public _CSRIteratorBase<MatrixPtrT> {
	public:
		/// Construct an iterator to the matrix m starting from currentRow and having column pointer columnPointer
		/// @param[in] m The matrix which the iterator will iterate
		/// @param[in] currentRow The row from which the iterator will start
		/// @param[in] columnPointer Index into the array which holds ms columns and values
		CSRIterator(
			MatrixPtrT m,
			const int currentRow,
			const int columnPointer
		) noexcept;

		/// Template constructor which handles copy constructions and construction of const iterator from non-const iterator
		/// @tparam Other The type of the matrix which the passed object iterates. We check if the pointer is pointer to const
		/// and if it is we can create only const iterators from it. If the type is just a pointer we can create const and 
		/// non-const iteratos with this constructor
		/// @param[in] other The iterator which will be copied when initializing this object
		template<typename Other, typename = std::enable_if_t<!is_ptr_to_const_t<Other> || is_ptr_to_const_t<MatrixPtrT>>>
		CSRIterator(const CSRIterator<Other>& other) :
			_CSRIteratorBase<MatrixPtrT>(other)
		{}

		CSRIterator& operator++() noexcept;
		CSRIterator operator++(int) noexcept;
		friend void swap(CSRIterator& a, CSRIterator& b) noexcept;
	};

	template<typename MatrixPtrT>
	inline CSRIterator<MatrixPtrT>::CSRIterator(
		MatrixPtrT m,
		const int currentRow,
		const int columnPointer
	) noexcept :
		_CSRIteratorBase<MatrixPtrT>(m, currentRow, columnPointer)
	{

	}

	template<typename MatrixPtrT>
	inline CSRIterator<MatrixPtrT>& CSRIterator<MatrixPtrT>::operator++() noexcept {
		const int currentColumnPointer = this->getColumnPointer() + 1;
		this->setColumnPointer(currentColumnPointer);
		int currentRow = this->getRow();
		assert(currentColumnPointer <= this->getRowStartPointer(currentRow + 1));
		while(currentRow < this->getDenseRowCount() && currentColumnPointer == this->getRowStartPointer(currentRow + 1)) {
			currentRow++;
		}
		this->setRow(currentRow);
		return *this;
	}

	template<typename MatrixPtrT>
	inline CSRIterator<MatrixPtrT> CSRIterator<MatrixPtrT>::operator++(int) noexcept {
		CSRIterator initialState = *this;
		++(*this);
		return initialState;
	}

	template<typename MatrixPtrT>
	inline void swap(CSRIterator<MatrixPtrT>& a, CSRIterator<MatrixPtrT>& b) noexcept {
		swap(a.currentElement, b.currentElement);
		std::swap(a.denseRowCount, b.denseRowCount);
	}

	
	template<typename MatrixPtrT>
	class CSRRowIterator : public _CSRIteratorBase<MatrixPtrT> {
	public:
		CSRRowIterator(
			MatrixPtrT m,
			const int currentStartIndex,
			const int currentPositionIndex
		) noexcept;

		template<typename Other, typename = std::enable_if_t<!is_ptr_to_const_t<Other> || is_ptr_to_const_t<MatrixPtrT>>>
		CSRRowIterator(const CSRRowIterator<Other>& other) :
			_CSRIteratorBase<MatrixPtrT>(other)
		{}

		CSRRowIterator& operator++() noexcept;
		CSRRowIterator operator++(int) noexcept;
		friend void swap(CSRRowIterator& a, CSRRowIterator& b) noexcept;
	};

	template<typename MatrixPtrT>
	inline CSRRowIterator<MatrixPtrT>::CSRRowIterator(
		MatrixPtrT m,
		const int currentStartIndex,
		const int currentPositionIndex
	) noexcept :
		_CSRIteratorBase<MatrixPtrT>(m, currentStartIndex, currentPositionIndex)
	{

	}

	template<typename MatrixPtrT>
	inline CSRRowIterator<MatrixPtrT>& CSRRowIterator<MatrixPtrT>::operator++() noexcept {
		const int row = this->getRow();
		const int nextRowStart = this->getRowStartPointer(row + 1);
		const int columnPointer = this->getColumnPointer() + 1;
		this->setColumnPointer(columnPointer);
		assert(columnPointer <= nextRowStart);
		if (columnPointer == nextRowStart) {
			this->setRow(row + 1);
		}
		return *this;
	}

	template<typename MatrixPtrT>
	inline CSRRowIterator<MatrixPtrT> CSRRowIterator<MatrixPtrT>::operator++(int) noexcept {
		CSRRowIterator initialState = *this;
		++(*this);
		return initialState;
	}

	template<typename MatrixPtrT>
	inline void swap(CSRRowIterator<MatrixPtrT>& a, CSRRowIterator<MatrixPtrT>& b) noexcept {
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
	template<typename T>
	class CSRMatrix {
	public:
		using Iterator = CSRIterator<CSRMatrix<T>*>;
		using ConstIterator = CSRIterator<const CSRMatrix<T>*>;

		using RowIterator = CSRRowIterator<CSRMatrix<T>*>;
		using ConstRowIterator = CSRRowIterator<const CSRMatrix<T>*>;

		using value_type = T;

		friend class _CSRIteratorBase<const CSRMatrix<T>*>;
		friend class _CSRIteratorBase<CSRMatrix<T>*>;

		CSRMatrix() noexcept;
		CSRMatrix(const TripletMatrix<T>& triplet) noexcept;

		CSRMatrix(const CSRMatrix&) = delete;
		CSRMatrix& operator=(const CSRMatrix&) = delete;

		CSRMatrix(CSRMatrix&&) noexcept = default;
		CSRMatrix& operator=(CSRMatrix&&) noexcept = default;

		~CSRMatrix() = default;

		int init(const TripletMatrix<T>& triplet) noexcept;
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

		// ********************************************************************
		// ************************ GENERAL ITERATOR **************************
		// ********************************************************************

		/// Iterator to the first element of the matrix
		Iterator begin() noexcept;
		/// @brief Iterator to one element past the end of the matrix
		/// It is undefined to dereference this iterator. Use it only in loop checks.
		Iterator end() noexcept;
		/// Constant iterator to the first element of the matrix
		ConstIterator begin() const noexcept;
		/// @brief Constant iterator to one element past the end of the matrix
		/// It is undefined to dereference this iterator. Use it only in loop checks.
		ConstIterator end() const noexcept;
		/// Constant iterator to the first element of the matrix
		ConstIterator cbegin() const noexcept;
		/// @brief Constant iterator to one element past the end of the matrix
		/// It is undefined to dereference this iterator. Use it only in loop checks.
		ConstIterator cend() const noexcept;

		// ********************************************************************
		// **************************** ROW ITERATOR **************************
		// ********************************************************************

		/// Return an iterator to the beggining of a row.
		/// @param[in] i The index of the row to which iterator will be given. Must be in range [0;denseRowCount]
		/// @returns Iterator to the i-th row
		RowIterator rowBegin(const int i) noexcept;
		/// Return an iterator one past the last element of a row. Dereferencing this iterator is undefined behavior
		/// @param[in] i The row which end iterator will be given. Must be [0;denseRowCount]
		/// @returns Iterator one past the last element of the i-th row. Dereferencing this is undefined behavior
		RowIterator rowEnd(const int i) noexcept;
		/// Return a constant iterator to the beggining of a row.
		/// @param[in] i The index of the row to which iterator will be given. Must be in range [0;denseRowCount]
		/// @returns Iterator to the i-th row
		ConstRowIterator rowBegin(const int i) const noexcept;
		/// Return a constant iterator one past the last element of a row. Dereferencing this iterator is undefined behavior
		/// @param[in] i The row which end iterator will be given. Must be [0;denseRowCount]
		/// @returns Iterator one past the last element of the i-th row. Dereferencing this is undefined behavior
		ConstRowIterator rowEnd(const int i) const noexcept;
		/// Return a constant iterator to the beggining of a row.
		/// @param[in] i The index of the row to which iterator will be given. Must be in range [0;denseRowCount]
		/// @returns Iterator to the i-th row
		ConstRowIterator crowBegin(const int i) const noexcept;
		/// Return a constant iterator one past the last element of a row. Dereferencing this iterator is undefined behavior
		/// @param[in] i The row which end iterator will be given. Must be [0;denseRowCount]
		/// @returns Iterator one past the last element of the i-th row. Dereferencing this is undefined behavior
		ConstRowIterator crowEnd(const int i) const noexcept;

		/// @brief Perform matrix vector multiplication out = A * mult
		/// Where A is the current CSR matrix
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		void rMult(const T* const mult, T* const out) const noexcept;
		/// @brief Perform matrix vector multiplication and addition: out = lhs + A * mult
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector which will be added to the matrix vector product A * mult
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		///< Makes sense only of multithreading is enabaled.
		void rMultAdd(const T* const lhs, const T* const mult, T* const out) const noexcept;
		/// @brief Perform matrix vector multiplication and subtraction: out = lhs - A * mult
		/// Where A is the current CSR matrix
		/// @param[in] lhs The vector from which the matrix vector product A * mult will be subtracted
		/// @param[in] mult The vector which will multiply the matrix to the right
		/// @param[out] out Preallocated vector where the result is stored
		/// @param[in] async Whether to launch the operation in async mode or wait for it to finish.
		///< Makes sense only of multithreading is enabaled.
		void rMultSub(const T* const lhs, const T* const mult, T* const out) const noexcept;

		/// Inplace multiplication by a scalar
		/// @param[in] scalar The scalar which will multiply the matrix
		void operator*=(const T scalar);
		/// Inplace addition of two CSR matrices.
		/// @param[in] other The matrix which will be added to the current one
		void inplaceAdd(const CSRMatrix<T>& other);
		/// Inplace subtraction of two CSR matrices.
		/// @param[in] other The matrix which will be subtracted from the current one
		void inplaceSubtract(const CSRMatrix<T>& other);

		/// If a non zero entry exists at position (row, col) set its value
		/// Does nothing if there is no nonzero entry at (row, col)
		/// @param[in] row The row of the entry which is going to be updated
		/// @param[in] col The column of the entry which is going to be updated
		/// @param[in] newValue The new value of the entry at (row, col)
		/// @returns True if the there is an entry at (row, col) and the update is successful
		bool updateEntry(const int row, const int col, const T newValue);
		/// Retrieve the value of the element at position (row, col)
		/// IMPORTANT: The getting element is NOT constant operation. Getting elements directly
		/// must be avoided.
		/// @param[in] row The row of the element
		/// @param[in] col The column of the element
		/// @returns The value of the element at position (row, col). Note 0 is possible result, but
		/// it does not mean that the element at (row, col) is imlicit (not in the sparse structure)
		T getValue(const int row, const int col) const;
		/// Sets all values to 0
		void zeroValues();
		/// If there is a value at position row, col adds value to it
		/// @param[in] row The row of the added element
		/// @param[in] col The column of the added element
		/// @param[in] value The value which will be added to element at position (row, col)
		bool addEntry(const int row, const int col, const T value);

		// ********************************************************************
		// ************************ PRECONDITIONERS ***************************
		// ********************************************************************

		/// Identity preconditioner. Does nothing, but implements the interface
		class IDPreconditioner {
			int apply([[maybe_unused]]const T* rhs, [[maybe_unused]]T* x) const noexcept {
				return 0;
			}
		};

		/// Symmetric Gauss-Seidel Preconditioner.
		class SGSPreconditioner {
		public:
			SGSPreconditioner(const CSRMatrix<T>& m) noexcept;
			SGSPreconditioner(const SGSPreconditioner&) = delete;
			SGSPreconditioner& operator=(const SGSPreconditioner&) = delete;
			SGSPreconditioner(SGSPreconditioner&&) noexcept = default;
			/// Apply the Symmetric Gauss-Seidel preconditioner to a vector
			/// @param[in] rhs Vector which will be preconditioned
			/// @param[out] x The result of preconditioning rhs
			/// @retval Non zero on error
			int apply(const T* rhs, T* x) const noexcept;
		private:
			const CSRMatrix<T>& m;
		};

		/// Zero fill in Incomplete LU Factorization
		class ILU0Preconditioner {
		public:
			ILU0Preconditioner(const CSRMatrix<T>& m) noexcept;
			ILU0Preconditioner(const ILU0Preconditioner&) = delete;
			ILU0Preconditioner& operator=(const ILU0Preconditioner&) = delete;
			ILU0Preconditioner(ILU0Preconditioner&&) noexcept = default;
			/// Apply the Symmetric Gauss-Seidel preconditioner to a vector
			/// @param[in] rhs Vector which will be preconditioned
			/// @param[out] x The result of preconditioning rhs
			/// @retval Non zero on error
			int apply(const T* rhs, T* x) const noexcept;
			int validate() noexcept;
		private:
			/// Uses the reference to the original matrix m in order to create the LU facroization
			/// The values for both L and U will be written in ilo0Val array, the ones whic usually
			/// appear on the main diagonal of L (or U) will not be written, we must keep in mind that they are there tough.
			int factorize() noexcept;
			/// Reference to the matrix for which this decomposition is made
			/// For this factorization the resulting matrix will have the same non zero pattern as the original matrix,
			/// Thus we will reuse the two arrays start and positions from the original matrix and allocate space only for the values
			const CSRMatrix<T>& m;
			/// The values array for the ILU0 preconditioner
			std::unique_ptr<T[]> ilu0Val;
		};

		/// Zero fill in Incomplete Cholesky preconditioner.
		/// This preconditioner can be applied only to symmetric positive definite matrices
		class IC0Preconditioner {
		public:
			IC0Preconditioner(const CSRMatrix<T>& m) noexcept;
			IC0Preconditioner(const IC0Preconditioner&) = delete;
			IC0Preconditioner& operator=(const IC0Preconditioner&) = delete;
			IC0Preconditioner(IC0Preconditioner&&) = default;
			/// This will do the actual factorization of the matrix passed to the class in the constructor
			int init() noexcept;
			/// @brief Apply the Incomplete Cholesky preconditioner to a vector.
			/// The procedure solves the system of equations M.x = rhs <=> x = Inverse(M).rhs
			/// Solving the system takes O(M.getDenseRowCount()) (time proportional to the number of dense)
			/// @param[in] rhs Vector which will be preconditioned
			/// @param[out] x The result of preconditioning rhs
			/// @retval Non zero on error
			int apply(const T* rhs, T* x) const noexcept;
		private:
			int factorize() noexcept;
			const CSRMatrix<T>& m;
			std::unique_ptr<T[]> ic0Val;
		};

		/// Factory function to generate preconditioners for this matrix
		/// @tparam precond Type of the preconditioner defined by the SolverPreconditioner enum
		/// @returns Preconditioner which can be used for this matrix
		template<SolverPreconditioner precond>
		decltype(auto) getPreconditioner() const noexcept;
	private:
		/// Array which will hold all nonzero entries of the matrix.
		/// This is of length  number of non-zero entries
		std::unique_ptr<T[]> values;
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
		int fillArrays(const TripletMatrix<T>& triplet) noexcept;
		/// Get the next row which has at least one element in it
		/// @param[in] currentStartIndex The current row
		/// @param[in] startLength The length of the start array (same as the number of rows)
		/// @returns The next non empty row
		int getNextStartIndex(int currentStartIndex, int startLength) const noexcept;
		/// @brief Generic function to which will perfrom out = op(lhs, A * mult).
		/// It is allowed out to be the same pointer as lhs.
		/// @tparam FunctorType type for a functor implementing operator(real* lhs, real* rhs)
		/// @param[in] lhs Left hand side operand.
		/// @param[in] mult The vector which will multiply the matrix (to the right).
		/// @param[out] out Preallocated vector where the result will be stored.
		/// @param[in] op Functor which will execute op(lhs, A * mult) it must take in two real vectors.
		template<typename FunctorType>
		void rMultOp(const T* const lhs, const T* const mult, T* const out, const FunctorType& op) const noexcept;

		/// Check if there is element at (row, col) and return the index in positions where the elements is
		/// if there is no such element return -1.
		/// @param[in] row The row of the element
		/// @param[in] col The column of the element
		/// @returns The index of the element in positions/values, or -1 if there is no element at (row, col)
		int getValueIndex(const int row, const int col) const;

		static T vectorMultFunctor([[maybe_unused]]const T lhs, const T rhs) {
			return rhs;
		}

		static_assert(std::is_convertible_v<CSRMatrix<T>::ConstIterator, CSRMatrix<T>::ConstIterator>);
		static_assert(std::is_convertible_v<CSRMatrix::Iterator, CSRMatrix::ConstIterator>);
		static_assert(! std::is_convertible_v<CSRMatrix::ConstIterator, CSRMatrix::Iterator>);
		static_assert(std::is_convertible_v<CSRMatrix::Iterator, CSRMatrix::Iterator>);
		static_assert(std::is_trivially_copy_constructible_v<CSRMatrix::ConstIterator>);
		static_assert(std::is_trivially_copy_constructible_v<CSRMatrix::Iterator>);

		static_assert(std::is_convertible_v<CSRMatrix::ConstRowIterator, CSRMatrix::ConstRowIterator>);
		static_assert(std::is_convertible_v<CSRMatrix::RowIterator, CSRMatrix::ConstRowIterator>);
		static_assert(! std::is_convertible_v<CSRMatrix::ConstRowIterator, CSRMatrix::RowIterator>);
		static_assert(std::is_convertible_v<CSRMatrix::RowIterator, CSRMatrix::RowIterator>);
		static_assert(std::is_trivially_copy_constructible_v<CSRMatrix::ConstRowIterator>);
		static_assert(std::is_trivially_copy_constructible_v<CSRMatrix::RowIterator>);

	};

	template<typename T>
	inline CSRMatrix<T>::CSRMatrix() noexcept :
		values(nullptr),
		positions(nullptr),
		start(nullptr),
		denseRowCount(0),
		denseColCount(0),
		firstActiveStart(0)
	{ }

	template<typename T>
	inline CSRMatrix<T>::CSRMatrix(const TripletMatrix<T>& triplet) noexcept :
		values(new T[triplet.getNonZeroCount()]),
		positions(new int[triplet.getNonZeroCount()]),
		start(new int[triplet.getDenseRowCount() + 1]),
		denseRowCount(triplet.getDenseRowCount()),
		denseColCount(triplet.getDenseColCount()),
		firstActiveStart(-1)
	{
		fillArrays(triplet);
	}

	template<typename T>
	inline int CSRMatrix<T>::init(const TripletMatrix<T>& triplet) noexcept {
		firstActiveStart = -1;
		denseRowCount = triplet.getDenseRowCount();
		denseColCount = triplet.getDenseColCount();
		const int nnz = triplet.getNonZeroCount();
		values.reset(new T[nnz]);
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

	template<typename T>
	inline int CSRMatrix<T>::getNonZeroCount() const noexcept {
		return start == nullptr ? 0 : start[getDenseRowCount()];
	}

	template<typename T>
	inline int CSRMatrix<T>::getDenseRowCount() const noexcept {
		return denseRowCount;
	}

	template<typename T>
	inline int CSRMatrix<T>::getDenseColCount() const noexcept {
		return denseColCount;
	}

	template<typename T>
	inline bool CSRMatrix<T>::hasSameNonZeroPattern(const CSRMatrix<T>& other) {
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

	template<typename T>
	inline typename CSRMatrix<T>::Iterator CSRMatrix<T>::begin() noexcept {
		return Iterator(this, firstActiveStart, 0);
	}
	
	template<typename T>
	inline typename CSRMatrix<T>::ConstIterator CSRMatrix<T>::begin() const noexcept {
		return cbegin();
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstIterator CSRMatrix<T>::cbegin() const noexcept {
		return ConstIterator(this, firstActiveStart, 0);
	}

	template<typename T>
	inline typename CSRMatrix<T>::Iterator CSRMatrix<T>::end() noexcept {
		const int columnPointer = getNonZeroCount();
		return Iterator(this, denseRowCount, columnPointer);
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstIterator CSRMatrix<T>::end() const noexcept {
		return cend();
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstIterator CSRMatrix<T>::cend() const noexcept {
		const int columnPointer = getNonZeroCount();
		return ConstIterator(this, denseRowCount, columnPointer);
	}

	template<typename T>
	inline typename CSRMatrix<T>::RowIterator CSRMatrix<T>::rowBegin(const int i) noexcept {
		assert(i < denseRowCount);
		return RowIterator(this, i, start[i]);
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstRowIterator CSRMatrix<T>::rowBegin(const int i) const noexcept {
		assert(i < denseRowCount);
		return ConstRowIterator(this, i, start[i]);
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstRowIterator CSRMatrix<T>::crowBegin(const int i) const noexcept {
		assert(i < denseRowCount);
		return ConstRowIterator(this, i, start[i]);
	}

	template<typename T>
	inline typename CSRMatrix<T>::RowIterator CSRMatrix<T>::rowEnd(const int i) noexcept {
		assert(i < denseRowCount);
		if(start[i] == start[i+1]) return rowBegin(i);
		return RowIterator(this, i + 1, start[i + 1]);
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstRowIterator CSRMatrix<T>::rowEnd(const int i) const noexcept {
		assert(i < denseRowCount);
		if(start[i] == start[i+1]) return rowBegin(i);
		return ConstRowIterator(this, i + 1, start[i + 1]);
	}

	template<typename T>
	inline typename CSRMatrix<T>::ConstRowIterator CSRMatrix<T>::crowEnd(const int i) const noexcept {
		assert(i < denseRowCount);
		if(start[i] == start[i+1]) return rowBegin(i);
		return ConstRowIterator(this, i + 1, start[i + 1]);
	}

	template<typename T>
	template<typename FunctorType>
	inline void CSRMatrix<T>::rMultOp(
		const T* const lhs,
		const T* const mult,
		T* const out,
		const FunctorType& op
	) const noexcept {
		auto iterateRows = [&](
			#ifdef SMM_MULTITHREADING
				const tbb::blocked_range<int>& range
			#endif
		) {
			#ifdef SMM_MULTITHREADING
				const int startRow = range.begin();
				const int endRow = range.end();
			#else
				const int startRow = 0;
				const int endRow = denseRowCount;
			#endif
			for(int row = startRow; row < endRow; row++) {
				while(start[row] == start[row + 1]) {
					out[row] = op(lhs[row], 0);
					row++;
					if(row == endRow) return;
				}
				T dot(0);
				for (int colIdx = start[row]; colIdx < start[row + 1]; ++colIdx) {
					const int col = positions[colIdx];
					const T val = values[colIdx];
					dot = _smm_fma(val, mult[col], dot);
				}
				out[row] = op(lhs[row], dot);
			}
		};
#ifdef SMM_MULTITHREADING
		tbb::parallel_for(tbb::blocked_range<int>(0, denseRowCount), iterateRows);
#else
		iterateRows();
#endif

	}

	template<typename T>
	inline void CSRMatrix<T>::rMult(const T* const mult, T* const res) const noexcept {
		assert(mult != res);
		rMultOp(res, mult, res, vectorMultFunctor);
	}

	template<typename T>
	inline void CSRMatrix<T>::rMultAdd(const T* const lhs, const T* const mult, T* const out) const noexcept {
		rMultOp(lhs, mult, out, [](const T lhs, const T rhs) { return lhs + rhs;});
	}

	template<typename T>
	inline void CSRMatrix<T>::rMultSub(const T* const lhs, const T* const mult, T* const out) const noexcept {
		rMultOp(lhs, mult, out, [](const T lhs, const T rhs) { return lhs - rhs;});
	}

	template<typename T>
	inline int CSRMatrix<T>::getNextStartIndex(int currentStartIndex, int startLength) const noexcept {
		do {
			currentStartIndex++;
		} while (currentStartIndex < startLength && start[currentStartIndex] == start[currentStartIndex + 1]);
		return currentStartIndex;
	}

	template<typename T>
	inline void CSRMatrix<T>::operator*=(const T scalar) {
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] *= scalar;
		}
	}

	template<typename T>
	inline void CSRMatrix<T>::inplaceAdd(const CSRMatrix<T>& other) {
		assert(hasSameNonZeroPattern(other) && "The two matrices have different nonzero patterns");
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] += other.values[i];
		}
	}

	template<typename T>
	inline void CSRMatrix<T>::inplaceSubtract(const CSRMatrix<T>& other) {
		assert(hasSameNonZeroPattern(other) && "The two matrices have different nonzero patterns");
		const int nonZeroCount = getNonZeroCount();
		for(int i = 0; i < nonZeroCount; ++i) {
			values[i] -= other.values[i];
		}
	}

	template<typename T>
	inline int CSRMatrix<T>::getValueIndex(const int row, const int col) const {
		assert(row >= 0 && row < denseRowCount);
		assert(col >= 0 && col < denseColCount);
		// Assumes that the columns are sorted in increasing order
		int rowBegin = start[row];
		int rowEnd = start[row + 1] - 1;
		while(rowBegin <= rowEnd) {
			const int mid = (rowBegin + rowEnd) / 2;
			const int currentColumn = positions[mid];
			if(col > currentColumn) {
				rowBegin = mid + 1;
			} else if(col < currentColumn) {
				rowEnd = mid - 1;
			} else {
				return mid;
			}
		}
		return -1;
	}

	template<typename T>
	inline bool CSRMatrix<T>::updateEntry(const int row, const int col, const T newValue) {
		const int index = getValueIndex(row, col);
		if(index != -1) {
			values[index] = newValue;
			return true;
		}
		return false;
	}

	template<typename T>
	inline T CSRMatrix<T>::getValue(const int row, const int col) const {
		const int index = getValueIndex(row, col);
		if(index != -1) {
			return values[index];
		}
		return T(0);
	}

	template<typename T>
	inline void CSRMatrix<T>::zeroValues() {
		std::fill_n(values.get(), getNonZeroCount(), T(0));
	}

	template<typename T>
	inline bool CSRMatrix<T>::addEntry(const int row, const int col, const T value) {
		const int index = getValueIndex(row, col);
		if(index == -1) {
			return 0;
		}
		values[index] += value;
		return 1;
	}

	template<typename T>
	inline int CSRMatrix<T>::fillArrays(const TripletMatrix<T>& triplet) noexcept {
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

	template<typename T>
	template<SolverPreconditioner precond>
	inline decltype(auto) CSRMatrix<T>::getPreconditioner() const noexcept {
		if constexpr (precond == SolverPreconditioner::NONE) {
			return IDPreconditioner();
		} else if constexpr (precond == SolverPreconditioner::SYMMETRIC_GAUS_SEIDEL) {
			return SGSPreconditioner(*this);
		}
	}

	template<typename T>
	inline CSRMatrix<T>::SGSPreconditioner::SGSPreconditioner(const CSRMatrix& m) noexcept :
		m(m)
	{}

	template<typename T>
	inline int CSRMatrix<T>::SGSPreconditioner::apply(const T* rhs, T* x) const noexcept {
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
			T value = m.values[indexInRow];
			T lhs = rhs[row];
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
			T value = m.values[indexInRow];
			T lhs(0);
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

	template<typename T>
	inline CSRMatrix<T>::ILU0Preconditioner::ILU0Preconditioner(const CSRMatrix<T>& m) noexcept : 
		m(m),
		ilu0Val(std::make_unique<T[]>(m.getNonZeroCount()))
	{	}

	template<typename T>
	inline int CSRMatrix<T>::ILU0Preconditioner::validate() noexcept {
		return factorize();
	}

	template<typename T>
	inline int CSRMatrix<T>::ILU0Preconditioner::factorize() noexcept {
		// L and U will have the same non zero pattern as the lower and upper triangular parts of m
		// The decompozition will take the form m = L * U + R, where we shall take only L and U and 
		// m - L * U will be zero for all non zero elements of m, but m - L * U might have some non zero
		// elements in places where m had zeros.

		const int rows = m.getDenseRowCount();
		const int cols = m.getDenseColCount();
		assert(rows == cols);
		memcpy(ilu0Val.get(), m.values.get(), sizeof(T) * m.getNonZeroCount());
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
		std::vector<int> columnIndex(cols, -1);
		// TODO [Move Diagonal To End]: This can be avoided if the diagonal elements are kept in a fixed position in each row
		// For example keep the diagonal element in the end of the row.
		Vector<T> diagonalElementsInv(rows);
		diagonalElementsInv[0] = T(1.0) / m.values[0];
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
				const T alphaIK = ilu0Val[kPos] * diagonalElementsInv[k];
				ilu0Val[kPos] = alphaIK;
				for(int colPos = m.start[k + 1] - 1, col = m.positions[colPos]; col > 0; col = m.positions[--colPos]) {
					const T betaKJ = ilu0Val[colPos];
					if(columnIndex[col] != -1) {
						ilu0Val[columnIndex[col]] -= alphaIK * betaKJ;
					}
				}
			}
			assert(k == row && ilu0Val[kPos] > 1e-6 && "Zero in pivot position!");
			if(k == row && ilu0Val[kPos] > 1e-6) {
				return 2;
			}
			diagonalElementsInv[k] = T(1.0) / ilu0Val[kPos];
			// Clear the column indexes and prepare them for the next iterations
			for(int i = rowStart; i < rowEnd; ++i) {
				const int column = m.positions[i];
				columnIndex[column] = -1;
			}
		}

		return 0;
	}

	template<typename T>
	inline CSRMatrix<T>::IC0Preconditioner::IC0Preconditioner(const CSRMatrix<T>& m) noexcept :
		m(m)
	{}

	template<typename T>
	inline int CSRMatrix<T>::IC0Preconditioner::init() noexcept {
		return factorize();
	}

	template<typename T>
	inline int CSRMatrix<T>::IC0Preconditioner::apply(const T* rhs, T* x) const noexcept {
		const int rows = m.getDenseRowCount();
		// Solve L.y = rhs, y = Transpose(L).x
		for(int row = 0; row < rows; ++row) {
			T sum = rhs[row];
			const int rowStart = m.start[row];
			const int rowEnd = m.start[row+1];
			int j = rowStart;
			int col = m.positions[j];
			while(col < row && j < rowEnd) {
				sum -= ic0Val[j] * x[col];
				j++;
				col = m.positions[j];
			}
			assert(col == row && "Missing diagonal element. This means that the original matrix was not SPD.");
			x[row] = sum / ic0Val[j];
		}

		// Solve Transpose(L).x = y
		for(int row = rows - 1; row >= 0; --row) {
			T sum = x[row];
			const int rowStart = m.start[row];
			const int rowEnd = m.start[row+1];
			int j = rowEnd - 1;
			int col = m.positions[j];
			while(col > row && j >= rowStart) {
				sum -= ic0Val[j] * x[col];
				j--;
				col = m.positions[j];
			}
			assert(col == row && "Missing diagonal element. This means that the original matrix was not SPD.");
			x[row] = sum / ic0Val[j];
		}
		return 0;
	}

	template<typename T>
	inline int CSRMatrix<T>::IC0Preconditioner::factorize() noexcept {
		const int nnz = m.getNonZeroCount();
		const int rows = m.getDenseRowCount();
		assert(rows == m.getDenseColCount());
		ic0Val.reset(new T[nnz]);
		// For each row this will be an offset to where we can put a value. For the i-th element we have that
		// nextFreeSlot[i] >= 0 && nextFreeSlot[i] < CSRMatrix::start[i + 1] - CSRMatrix::start[i] i.e. the i-th elements is
		// between 0 and the number of nonzero elements in the row. In order to obtain the correcto position
		// of the next nonzero element of the row we must compute: CSRMatrix::start[i] + nextFreeSlot[i]
		std::vector<int> nextFreeSlot(rows, 0);
		// This will hold the index into CSRMatrix::positions for each nonzero column of the row in the most outer loop below,
		// so that CSRMatrix::positions[usedColumns[c]] will be the same as c. We can use CSRMatrix::positions[usedColumns[c]] to
		// find the value of the element at that particular column of the row in the most outer loop.
		std::vector<int> usedColumns(rows, -1);
		// The outer loop goes trough all rows of the matrix. After each iteration of the loop all elements of the form
		// l_j,i for j = i...rows will be found.
		for(int i = 0; i < rows; ++i) {
			// For each value we shall multiply the current row by some other row .we want to multiply only non-zero elements.
			// Since we shall iterate non-zero elements of the "other" row, we need to mark which of the elements in the current
			// (main) row are non zero. We do this by putting the index in CSRMatrix::values to the value at that particular column
			for(int j = m.start[i]; j < m.start[i+1]; ++j) {
				const int col = m.positions[j];
				usedColumns[col] = j;
			}
			// Use separate loop to handle the diagonal element
			T diagonalElement(0);
			int columnIndex = m.start[i];
			int column = m.positions[columnIndex];
			while(column < i) {
				diagonalElement += ic0Val[columnIndex] * ic0Val[columnIndex];
				columnIndex++;
				column = m.positions[columnIndex];
			}
			if(column != i) {
				assert(false && "The matrix is not positive definite");
				return 1;
			}
			const int diagonalPosition = m.start[i] + nextFreeSlot[i];
			assert(m.values[columnIndex] - diagonalElement > 0 && "The matrix is not positive definite");
			diagonalElement = std::sqrt(m.values[columnIndex] - diagonalElement);
			assert(std::abs(diagonalElement) > 1e-6 && "The diagonal element is 0. The matrix is not possitive definite.");
			ic0Val[diagonalPosition] = diagonalElement;
			nextFreeSlot[i]++;
			const T diagonalIversed = T(1) / diagonalElement;
			
			// When we represent the matrix in the form L*Transpose(L) for the element at position (i, j)
			// of the original matrix is given by: a_i,j = Sum(l_i,k * l_k,j). Because of the symmetry of the matrix
			// this is the same as: a_j_i = Sum(l_i,k * l_j,k). 
			for(int j = i + 1; j < rows; ++j) {
				const int rowStart = m.start[j];
				const int valueIndex = rowStart + nextFreeSlot[j];
				// This iteration of the loop seeks to find l_j,i but since we are doing incomplete Cholesky factorization
				// we drop all elements l_j,i for which the corresponding element in the original matrix m_j,i is zero
				if(m.positions[valueIndex] != i) {
					continue;
				}
				// The most inner loop of the tree is just doing the sum: Sum(l_i,k * l_j,k) for k < i 
				T sum(0);
				const int rowEnd = m.start[j+1];
				int k = rowStart, column = m.positions[k];
				while(k < rowEnd && column < i) {
					const int iValueIndex = usedColumns[column];
					if(iValueIndex != -1) {
						sum += ic0Val[iValueIndex] * ic0Val[k];
					}
					k++;
					column = m.positions[k];
				}
				// After the while loop finishes k will be an index into CSRMatrix::values/positions to an element
				// whose row index is grater than i. The symmetric element to the one we have just found is situated
				// at some previous row with row index smaller than i. So we put it in two possitions:
				// 1) k which will belongs to the lower triangular matrix and is sutuated towards the end of CSRMatrix::values
				// 2) ic0Val[valueIndex] which belongs to the upper triangular matrix and is situated towards the beggining of
				// CSRMatrix::values
				sum = (m.values[k] - sum) * diagonalIversed;
				ic0Val[k] = sum;
				const int uppertTriangularPosition = m.start[i] + nextFreeSlot[i];
				ic0Val[uppertTriangularPosition] = sum;
				nextFreeSlot[i]++;
				nextFreeSlot[j]++;
			}
			// Reset the state of the used columns
			for(int j = m.start[i]; j < m.start[i+1]; ++j) {
				const int col = m.positions[j];
				usedColumns[col] = -1;
			}
		}
		return 0;
	}

	template<typename T>
	inline void saveDenseText(const char* filepath, const CSRMatrix<T>& m) {
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
		typename CSRMatrix<T>::ConstIterator it = m.begin();
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
	inline void toLinearDenseRowMajor(const CompressedMatrixFormat& compressed, typename CompressedMatrixFormat::value_type* out) noexcept {
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
	template<typename T>
	inline SolverStatus BiCGSymmetric(
		const CSRMatrix<T>& a,
		T* b,
		T* x,
		int maxIterations,
		T eps
	) {

		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		Vector<T> r(a.getDenseRowCount());
		a.rMultSub(b, x, r);

		Vector<T> p(a.getDenseRowCount());
		std::copy_n(r.begin(), a.getDenseRowCount(), p.begin());

		Vector<T> ap(a.getDenseRowCount());

		T rSquare = r * r;
		int iterations = 0;
		const T epsSquared = eps * eps;
		const int rows = a.getDenseRowCount();
		do {
			a.rMult(p, ap);
			const T denom = ap* p;
			// Numerical instability will cause devision by zero (or something close to). The method must be restarted
			// For positive definite matrices if denom becomes 0 this is a lucky breakdown so we should not exit with error
			// but continue iterating. However we cannot know in advance if the matrix is positive definite, thus a heuristic is used.
			// If a system with positive definite matrix is solved, near the lucky breakdown the residual must be small, so it's length
			// squared will be small too, so for rSquare and small denom we continue, if the rSquare is large we are most likely dealing
			// with indefinite matrix and this is serious breakdown so we exit the procedure with error message.
			if (eps > std::abs(denom) && rSquare > 1) {
				return SolverStatus::DIVERGED;
			}
			const T alpha = rSquare / denom;
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
				for (int j = range.begin(); j < range.end(); ++j) {
					x[j] += alpha * p[j];
					r[j] -= alpha * ap[j];
				}
			});
#else
			for(int i = 0; i < rows; ++i) {
				x[i] += alpha * p[i];
				r[i] -= alpha * ap[i];
			}
#endif
			// Dot product r * r can be zero (or close to zero) only if r has length close to zero.
			// But if the residual is close to zero, this means that we have found a solution
			const T newRSquare = r * r;
			// If rSquare is small it's expected next iteration residual to be small too
			// Thus deleting large number by a small is highly unlikely here
			// If rSquare is small and newRSquare is large, we have critical brakedown, which might happen with BiCG method
			if(newRSquare > 1 && rSquare < eps) {
				return SolverStatus::DIVERGED;
			}
			const T beta = newRSquare / rSquare;
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
				for (int j = range.begin(); j < range.end(); ++j) {
					p[j] = r[j] + beta * p[j];
				}
			});
#else
			for(int i = 0; i < rows; ++i) {
				p[i] = r[i] + beta * p[i];
			}
#endif
			rSquare = newRSquare;
			iterations++;
		} while (rSquare > epsSquared && iterations < maxIterations);

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
	template<typename T>
	inline SolverStatus ConjugateGradientSquared(const CSRMatrix<T>& a, T* b, T* x, int maxIterations, T eps) {
		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		const int rows = a.getDenseRowCount();
		Vector<T> r(rows), r0(rows);
		a.rMultSub(b, x, r);
		
		// Help vectors, as in Saad's book, the vectors in the polynomial reccursion are: q, p, r
		// They can be expressed only in terms of themselves, the other vectors do same some computation
		// of the use a lot of memory they can be removed.
		Vector<T> p(rows), u(rows), q(rows), alphaUQ(rows), ap(rows);
		std::copy_n(r.begin(), rows, p.begin());
		std::copy_n(r.begin(), rows, u.begin());
		std::copy_n(r.begin(), rows, r0.begin());

		T rr0 = r * r0;
		int iterations = 0;
		const T epsSquared = eps * eps;
		do {
			a.rMult(p, ap);
			const T denom = ap * r0;
			// Must investigate if denom < eps is critical breakdown
			const T alpha = rr0 / denom;
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
				for (int j = range.begin(); j < range.end(); ++j) {
					q[j] = _smm_fma(-alpha, ap[j], u[j]);
					alphaUQ[j] = alpha * (u[j] + q[j]);
					x[j] = x[j] + alphaUQ[j];
				}
			});
#else
			for(int i = 0; i < rows; ++i) {
				q[i] = _smm_fma(-alpha, ap[i], u[i]);
				alphaUQ[i] = alpha * (u[i] + q[i]);
				x[i] = x[i] + alphaUQ[i];
			}
#endif
			a.rMultSub(r, alphaUQ, r);
			const T newRR0 = r * r0;
			// Must investigate if rr0 < eps is critical breakdown
			const T beta = newRR0 / rr0;
			
#if SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows), [&](const tbb::blocked_range<int>& range) {
				for (int j = range.begin(); j < range.end(); ++j) {
					u[j] = _smm_fma(beta, q[j], r[j]);
					p[j] = _smm_fma(beta, _smm_fma(beta, p[j], q[j]), u[j]);
				}
			});
#else
			for(int i = 0; i < rows; ++i) {
				u[i] = _smm_fma(beta, q[i], r[i]);
				p[i] = _smm_fma(beta, _smm_fma(beta, p[i], q[i]), u[i]);
			}
#endif
			rr0 = newRR0;
			iterations++;
			const T residualSquared = r * r;
		} while (residualSquared > epsSquared && iterations < maxIterations);

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
	template<typename Preconditioner, typename T>
	inline SolverStatus BiCGStab(
		const CSRMatrix<T>& a,
		T* b,
		T* x,
		int maxIterations,
		T eps,
		const Preconditioner& preconditioner
	) {
		maxIterations = std::min(maxIterations, a.getDenseRowCount());
		if (maxIterations == -1) {
			maxIterations = a.getDenseRowCount();
		}

		const int rows = a.getDenseRowCount();
		// This vector is allocated only if there is some preconditioner different than the identity
		// It is used to store intermediate data needed by the preconditioner
		Vector<T> precondScratchpad;
		constexpr bool precondition = !std::is_same<Preconditioner, decltype(a.template getPreconditioner<SolverPreconditioner::NONE>())>::value;
		if constexpr(precondition) {
			precondScratchpad.init(rows);
		}

		Vector<T> r(rows), r0(rows), p(rows), ap(rows), s(rows), as(rows);
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

		T resL2Norm = T(0);
		int iterations = 0;
		T rr0 = r * r0;
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

			T denom = ap * r0;
			const T alpha = rr0 / denom;
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
			const T omega = (as * s) / denom;
			resL2Norm = T(0);
			for(int i = 0; i < rows; ++i) {
				x[i] = _smm_fma(alpha, p[i], _smm_fma(omega, s[i], x[i])); 
				r[i] = _smm_fma(-omega, as[i], s[i]); 
				resL2Norm += r[i] * r[i];
			}
			resL2Norm = std::sqrt(resL2Norm);
			const T newRR0 = r * r0;
			// TODO: add proper check for division by zero
			const T beta = (newRR0 * alpha) / (rr0 * omega);
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
	template<typename T>
	inline SolverStatus BiCGStab(
		const CSRMatrix<T>& a,
		T* b,
		T* x,
		int maxIterations,
		T eps
	) {
		return BiCGStab(a, b, x, maxIterations, eps, a.template getPreconditioner<SolverPreconditioner::NONE>());
	}

	///@brief Solve a.x=b using Cojugate Gradient method
	/// Matrix a should be symmetric positive definite matrix. 
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in] x0 Initial condition
	/// @param[out] x Output vector (could be the same as the initial condition)
	/// @param[in] maxIterations Iterations threshold for the method.
	/// If convergence was not reached for less than maxIterations the method will exit.
	/// If maxIterations is -1 the method will do all possible iterations (the same as the number of rows in the matrix)
	/// @param[in] eps Required size of the L2 norm of the residual
	/// @return SolverStatus the status the solved system
	template<typename T>
	inline SolverStatus ConjugateGradient(
		const CSRMatrix<T>& a,
		const T* const b,
		const T* const x0,
		T* const x,
		int maxIterations,
		T eps
	) {
		// The algorithm in pseudo code is as follows:
		// 1. r_0 = b - A.x_0
		// 2. p_0 = r_0
		// 3. for j = 0, j, ... until convergence/max iteratoions
		// 4.	alpha_i = (r_j, r_j) / (A.p_j, p_j)
		// 5.	x_{j+1} = x_j + alpha_j * p_j
		// 6.	r_{j+1} = r_j - alpha_j * A.p_j
		// 7. 	beta_j = (r_{j+1}, r_{j+1}) / (r_j, r_j)
		// 8.	p_{j+1} = r_{j+1} + beta_j * p_j
		const int rows = a.getDenseRowCount();
		const T epsSuared = eps * eps;
		Vector<T> r(rows, T(0));
		a.rMultSub(b, x0, r);

		Vector<T> p(rows), Ap(rows, T(0));
		std::copy(r.begin(), r.end(), p.begin());
		T residualNormSquared = r * r;
		if(epsSuared > residualNormSquared) {
			return SolverStatus::SUCCESS;
		}
		if(maxIterations == -1) {
			maxIterations = rows;
		}
		// We have initial condition different than the output vector on the first iteration when we compute
		// x = x + alpha * p, we must have the x on the right hand side to be the initial condition x. And on all
		// next iterations it must be the output vector.
		const T* currentX = x0;
		for(int i = 0; i < maxIterations; ++i) {
			a.rMult(p, Ap);
			const T pAp = Ap * p;
			// If the denominator is 0 we have a lucky breakdown. The residual at the previous step must be 0.
			assert(pAp != 0);
			// alpha = (r_i, r_i) / (Ap, p)
			const T alpha = residualNormSquared / pAp;
			// x = x + alpha * p
			// r = r - alpha * Ap
			T newResidualNormSquared = 0;
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows),[&](const tbb::blocked_range<int>& range) {
				for(int j = range.begin(); j < range.end(); ++j) {
					x[j] = _smm_fma(alpha, p[j], currentX[j]);
					r[j] = _smm_fma(-alpha, Ap[j], r[j]);
				}
			});
			newResidualNormSquared = r * r;
#else
			for(int j = 0; j < rows; ++j) {
				x[j] = _smm_fma(alpha, p[j], currentX[j]);
				r[j] = _smm_fma(-alpha, Ap[j], r[j]);
				newResidualNormSquared += r[j] * r[j];
			}
#endif
			if(epsSuared > newResidualNormSquared) {
				return SolverStatus::SUCCESS;
			}
			// beta = (r_{i+1}, r_(i+1)) / (r_i, r_i)
			const T beta = newResidualNormSquared / residualNormSquared;
			residualNormSquared = newResidualNormSquared;
			// p = r + beta * p
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows),[&](const tbb::blocked_range<int>& range){
				for(int j = range.begin(); j < range.end(); ++j) {
					p[j] = _smm_fma(beta, p[j], r[j]);
				}
			});
#else
			for(int j = 0; j < rows; ++j) {
				p[j] = _smm_fma(beta, p[j], r[j]);
			}
#endif
			currentX = x;
		}
		return SolverStatus::MAX_ITERATIONS_REACHED;
	}

	/// Preconditioned version of the Conjugate Gradient method. With Incomplete Cholesky preconditioner
	/// The preconditioner is given in the form L*Transpose(L). This is a separate algorithm, since IC0,
	/// is symmetric and we can utilize this property in the solver
	/// Matrix a should be symmetric positive definite matrix. 
	/// @param[in] a Coefficient matrix for the system of equations
	/// @param[in] b Right hand side for the system of equations
	/// @param[in] x0 Initial condition
	/// @param[out] x Output vector (could be the same as the initial condition)
	/// @param[in] maxIterations Iterations threshold for the method.
	/// If convergence was not reached for less than maxIterations the method will exit.
	/// If maxIterations is -1 the method will do all possible iterations (the same as the number of rows in the matrix)
	/// @param[in] eps Required size of the L2 norm of the residual
	/// @param[in] preconditioner Incomplete Cholesky preconditioner which will be used precondition this system.
	/// @return SolverStatus the status the solved system
	template<typename T>
	inline SolverStatus ConjugateGradient(
		const CSRMatrix<T>& a,
		const T* const b,
		const T* const x0,
		T* const x,
		int maxIterations,
		T eps,
		const typename CSRMatrix<T>::IC0Preconditioner& M
	) {
		// Pseudo code for the algorithm:
		// 1. r_0 = b - A.x_0
		// 2. z_0 = Inverse(M).r_0
		// 3. p_0 = z_0
		// 4. for j = 0, 1, ... until convergence/max iterations
		// 5.	alpha_j = (r_j, z_j) / (A.p_j, p_j)
		// 6.	x_{j+1} = x_j + alpha_j * p_j
		// 7.	r_{j+1} = r_j - alpha_j * A.p_j
		// 8.	z_{j+1} = Inverse(M).r_{j+1}
		// 9.	beta_j = (r_{j+1}, z_{j+1}) / (r_j, z_j)
		// 10.	p_{j+1} = z_{j+1} + beta_j * p_j
		const int rows = a.getDenseRowCount();
		const T epsSuared = eps * eps;
		Vector<T> r(rows, 0);
		Vector<T> z(rows, 0);
		Vector<T> p(rows, 0);
		a.rMultSub(b, x0, r);
		M.apply(r, z);
		T rz = 0;
		T residualNormSquared = 0;
		for(int i = 0; i < rows; ++i) {
			rz += r[i] * z[i];
			residualNormSquared += r[i] * r[i];
			p[i] = z[i];
		}
		if(epsSuared > residualNormSquared) {
			return SolverStatus::SUCCESS;
		}
		if(maxIterations == -1) {
			maxIterations = rows;
		}
		Vector<T> Ap(rows, 0);
		// We have initial condition different than the output vector on the first iteration when we compute
		// x = x + alpha * p, we must have the x on the right hand side to be the initial condition x. And on all
		// next iterations it must be the output vector.
		const T* currentX = x0;
		for(int i = 0; i < maxIterations; ++i) {
			a.rMult(p, Ap);
			const T pAp = Ap * p;
			// If the denominator is 0 we have a lucky breakdown. The residual at the previous step must be 0.
			assert(pAp != 0);
			// alpha_j = (r_j, z_j) / (A.p_j, p_j)
			const T alpha = rz / pAp;
			// x_{j+1} = x_j + alpha_j * p_j
			// r_{j+1} = r_j - alpha_j * A.p_j
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows),[&](const tbb::blocked_range<int>& range) {
				for(int j = range.begin(); j < range.end(); ++j) {
					x[j] = _smm_fma(alpha, p[j], currentX[j]);
					r[j] = _smm_fma(-alpha, Ap[j], r[j]);
				}
			});
#else
			for(int j = 0; j < rows; ++j) {
				x[j] = _smm_fma(alpha, p[j], currentX[j]);
				r[j] = _smm_fma(-alpha, Ap[j], r[j]);
			}
#endif
			M.apply(r, z);
			T newRZ = r * z;
			residualNormSquared = r * r;
			if(epsSuared > residualNormSquared) {
				return SMM::SolverStatus::SUCCESS;
			}
			const T beta = newRZ / rz;
			// p_{j+1} = z_{j+1} + beta_j * p_j
#ifdef SMM_MULTITHREADING
			tbb::parallel_for(tbb::blocked_range<int>(0, rows),[&](const tbb::blocked_range<int>& range) {
				for(int j = range.begin(); j < range.end(); ++j) {
					p[j] = _smm_fma(beta, p[j], z[j]);
				}
			});
#else
			for(int j = 0; j < rows; ++j) {
				p[j] = _smm_fma(beta, p[j], z[j]);
			}
#endif
			rz = newRZ;
			currentX = x;
		}
		return SMM::SolverStatus::MAX_ITERATIONS_REACHED;
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
	template<typename T>
	inline MatrixLoadStatus loadMatrixMarketMatrix(const char* filepath, TripletMatrix<T>& out) {
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
			T value;
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

	template<typename T>
	inline MatrixLoadStatus loadSMMDTMatrix(const char* filepath, TripletMatrix<T>& out) {
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
				T val;
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

	template<typename T>
	inline MatrixLoadStatus loadMatrix(const char* filepath, TripletMatrix<T>& out) {
		const char* fileExtension = strrchr(filepath, '.') + 1;
		if (strcmp(fileExtension, "mtx") == 0) {
			return loadMatrixMarketMatrix(filepath, out);
		} else if (strcmp(fileExtension, "smmdt") == 0) {
			return loadSMMDTMatrix(filepath, out);
		} else {
			return MatrixLoadStatus::FAILED_TO_OPEN_FILE_UNKNOWN_FORMAT;
		}
	}

	template<typename T>
	inline MatrixLoadStatus loadMatrix(const char* filepath, CSRMatrix<T>& out) {
		SMM::TripletMatrix<T> triplet;
		const MatrixLoadStatus status = loadMatrix(filepath, triplet);
		if (status != MatrixLoadStatus::SUCCESS) {
			return status;
		}
		out.init(triplet);
		return MatrixLoadStatus::SUCCESS;
	}
}
