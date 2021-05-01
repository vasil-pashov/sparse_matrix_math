#pragma once
#include "sparse_matrix_math.h"

static constexpr inline SMM::real L2Epsilon() {
	return SMM::real(1e-6);
}

static constexpr inline SMM::real MaxInfEpsilon() {
	return SMM::real(1e-4);
}