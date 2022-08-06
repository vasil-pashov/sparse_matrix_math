# sparse_matrix_math
Header only, C++17 library for solving systems of equations with sparse matrices. Provides [iterative methods](#Iterative-Methods) for solving systems of linear equations. Matrices can be represented in [triplet/coordinate format](https://en.wikipedia.org/wiki/Sparse_matrix#Coordinate_list_(COO)) and [compressed sparse row format](https://en.wikipedia.org/wiki/Sparse_matrix#Compressed_sparse_row_(CSR,_CRS_or_Yale_format)). There is support for [preconditioned iterations](#Preconditioned-Iterations) for some of the implemented methods. Methods can be [run in parallel](#Parallel-Implementation). Limited support for loading [matrix market](#Matrix-Market) files is provided.

# Iterative Methods
Currently available methods are Krylov iterative methods derivatives of the BiConjugate Gradient Method. These methods can work with positive definite, negative definite and indefinite matrices. In case of symmetric positive definite and symmetric negative definite matrix the methods will converge to the exact answer. Theoretically speaking given SPD or SND matrices the methods will find the exact answer after no more that `m` steps where `m` is the size of the matrix. In the case of indefinite matrix the methods can occasionally diverge. In that case the method must be restarted with different initial guess. Keep in mind that there is no Krylov method which is proven to converge for general indefinite matrix.

## Conjugate Gradient Method
This method is proven to converge for every SPD matrix. Keep in mind that the proof is with exact math (floating point arithmetics could lead to divergence).
```cpp
SMM::CSRMatrix m;
// Fill m
...
// Init the right hand side somehow
float* rhs = initRhs()
// Initial guess.
float* x0 = initialGuess();
// The result will be stored here. The vector must be preallocated by the caller.
float* res = new float[m.getDenseRowCount()];
// The method will do no more than maxIterations iterations. If maxIterations is -1 the method will use the number of rows of the matrix as stopping condition.
const int maxIterations = 100;
// This will be used by the method as a condition for convergence. If the second norm of the residual becomes smaller, the method will end.
// Note it is allowed for x0 and res to be the same vector. If this happens x0 will be overriten.
const float L2NormCondition = 1e-6;
SMM::SolverStatus status = SMM::ConjugateGradient(m, rhs, x0, res, maxIterations, L2NormCondition);
```

## BiConjugate Gradient Symmetric Method
This is a variant of the BiConjugate Gradient, where the input matrix is known to be symmetric. For SPD matrix would yeald the exactly the same result as the Conjugate Gradient method. Example call:
```cpp
SMM::CSRMatrix m;
// Fill m
...
// Init the right hand side somehow
float* rhs = initRhs()
// The result will be stored here. The vector must be preallocated by the caller.
float* res = new float[m.getDenseRowCount()];
// The method will do no more than maxIterations iterations. If maxIterations is -1 the method will use the number of rows of the matrix as stopping condition.
const int maxIterations = 100;
// This will be used by the method as a condition for convergence. If the second norm of the residual becomes smaller, the method will end.
const float L2NormCondition = 1e-6;
SMM::SolverStatus status = SMM::BiCGSymmetric(m, rhs, res, maxIterations, L2NormCondition);
```
## Conjugate Gradient Squared Method
Transpose free variation of the BiConjugate Gradient method. Can be used with general matricies. In practice, the method usually converges twice as fast as the BiCG method, but the squaring of the underlying residual polynomial makes the method more susceptible to rounding errors.
```cpp
SMM::CSRMatrix m;
// Fill m
...
// Init the right hand side somehow
float* rhs = initRhs()
// The result will be stored here. The vector must be preallocated by the caller.
float* res = new float[m.getDenseRowCount()];
// The method will do no more than maxIterations iterations. If maxIterations is -1 the method will use the number of rows of the matrix as stopping condition.
const int maxIterations = 100;
// This will be used by the method as a condition for convergence. If the second norm of the residual becomes smaller, the method will end.
const float L2NormCondition = 1e-6;
SMM::SolverStatus status = SMM::ConjugateGradientSqared(m, rhs, res, maxIterations, L2NormCondition);
```
## BiConjugate Gradient Stabilized
Another transpose free variant of the BiConjugate Gradient Method which can be used on general matrices. This method does not square the residual polynomial, but uses polynomial product which smoothens the convergence behavior.
```cpp
SMM::CSRMatrix m;
// Fill m
...
// Init the right hand side somehow
float* rhs = initRhs()
// The result will be stored here. The vector must be preallocated by the caller.
float* res = new float[m.getDenseRowCount()];
// The method will do no more than maxIterations iterations. If maxIterations is -1 the method will use the number of rows of the matrix as stopping condition.
const int maxIterations = 100;
// This will be used by the method as a condition for convergence. If the second norm of the residual becomes smaller, the method will end.
const float L2NormCondition = 1e-6;
SMM::SolverStatus status = SMM::BiCGStab(m, rhs, res, maxIterations, L2NormCondition);
```
For preconditioned iterations check [Preconditioners](#preconditioners)

# Preconditioners
Preconditioned iterations are allowed only for the [BiConjugate Gradient Stabilized](#biconjugate-gradient-stabilized). Preconditioners are generated by `SMM::CSRMatrix::getPreconditioner(SMM::SolverPreconditioner)` and then passed to `BiCGStab`. Example usage:
 ```cpp
SMM::CSRMatrix m;
// Fill m
...
// Init the right hand side somehow
float* rhs = initRhs()
// The result will be stored here. The vector must be preallocated by the caller.
float* res = new float[m.getDenseRowCount()];
// The method will do no more than maxIterations iterations. If maxIterations is -1 the method will use the number of rows of the matrix as stopping condition.
const int maxIterations = 100;
// This will be used by the method as a condition for convergence. If the second norm of the residual becomes smaller, the method will end.
const float L2NormCondition = 1e-6;
SMM::SolverStatus status = SMM::BiCGStab(m, rhs, res, maxIterations, L2NormCondition, preconditioner, m.getPreconditioner(SMM::SolverPreconditioner::SYMMETRIC_GAUSS_SEIDEL));
```
## Symmetric Gauss-Seidel
Static preconditioner which does not take additional time to prepare, nor does it take additional space. It takes the form of `M = (D + L)inv(D)(D + U)` where `D`, `L` and `U` are the diagonal, lower triangular and upper triangular portions of the matrix which will be predonditioned.

# Dependencies
## Handling
When possible dependencies are handled via CMake FetchContent feature. Currently all dependencies are handled with FetchContent. The order is the following:

1. CMake will first try to find the dependencies via find_package
2. If find_package fails to find a dependency the source code will be downloaded (at configure time) and will be added (as subdirectory) to the project.

In general it is recommended to build and install the dependecies independently of the project and the pass `CMAKE_PREFIX_PATH`.

### Known issues
The CMake configuration for TBB does not provide an option to turn off the INSTALL target https://github.com/oneapi-src/oneTBB/pull/800. When TBB is added as a subdirectory it will always generate install target and will interfere with the install target of the project.

## List
* Doctest v2.4.8
* TBB v2021.5.0

# Build and Run Tests
```sh
git clone https://github.com/vasil-pashov/sparse_matrix_math.git
cd sparse_matrix_math
mkdir build
cmake -B"./build" -DCMAKE_BUILD_TYPE=Release -DSMM_WITH_TESTS=ON
cd build
cmake --build . --config Release
ctest
```

# Install
The library can be installed to the system, so that other cmake project can use it via the `fing_package` utility.
```sh
git clone https://github.com/vasil-pashov/cpp_tm.git
cd cpp_tm
mkdir build
cmake -B"./build" -DCMAKE_BUILD_TYPE=Release -DCPPTM_UNIT_TESTS=OFF
cd build
sudo cmake --install ./
```

To change the install directories use `CMAKE_INSTALL_PREFIX` and `CONFIG_INSTALL_DIR` variables (if some nonstandard structure is needed). Where `CONFIG_INSTALL_DIR` gets appended to `CMAKE_INSTALL_PREFIX`.

# CMake Options
* SMM_WITH_MULTITHREADING - Enables multithreading for the library. This will create dependency on the TBB library. Default `ON`.
* SMM_WITH_TESTS - If set this will build the tests project. Default `OFF`.
* SMM_WITH_INSTALL - If this is set to true an install target will be generated. Default `OFF`.
