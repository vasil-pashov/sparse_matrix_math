#pragma once
#include "sparse_matrix_math.h"
#include "doctest/doctest.h"

#define DOCTEST_VALUE_PARAMETERIZED_DATA(data, data_container)                                  \
    static size_t _doctest_subcase_idx = 0;                                                     \
    std::for_each(data_container.begin(), data_container.end(), [&](const auto& in) {           \
        DOCTEST_SUBCASE((std::string(#data_container "[") +                                     \
                        std::to_string(_doctest_subcase_idx++) + "]").c_str()) { data = in; }  \
    });                                                                                         \
    _doctest_subcase_idx = 0

template<typename T>
inline SMM::Vector<T> sumColumsPerRow(const SMM::CSRMatrix<T>& m) {
	const int numRows = m.getDenseRowCount();
	SMM::Vector<T> v(numRows, 0);
	for (const auto& el : m) {
		v[el.getRow()] += el.getValue();
	}
	return v;
}

inline std::string getMatrixPath(const std::string& name) {
	return ASSET_PATH + name;
}

template<typename T>
inline constexpr T l2Eps();

template<>
inline constexpr float l2Eps() {
	return 1e-4;
}

template<>
inline constexpr double l2Eps() {
	return 1e-8;
}

template<typename T>
inline constexpr T infEps();

template<>
inline constexpr float infEps() {
	return 1e-4;
}

template<>
inline constexpr double infEps() {
	return 1e-8;
}