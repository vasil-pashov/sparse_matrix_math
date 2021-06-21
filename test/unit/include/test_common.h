#pragma once
#include "sparse_matrix_math.h"

template<typename T>
static constexpr inline T L2Epsilon() {
	return T(1e-8);
}

template<typename T>
static constexpr inline T MaxInfEpsilon() {
	return T(1e-4);
}