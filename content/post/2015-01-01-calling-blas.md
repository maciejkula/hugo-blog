---
layout: post
title: Calling BLAS from Cython
excerpt: "Calling BLAS routines directly from Cython is easy to accomplish and can be extrememly useful."
tags: [Cython, BLAS]
modified: 2015-01-01
comments: true
---

It is often useful to be able to call [BLAS](http://en.wikipedia.org/wiki/Basic_Linear_Algebra_Subprograms) routines directly from Cython. Doing so avoids calling the corresponding NumPy functions (which would incur a performance penalty of running interpreted code and type and shape checking) as well as re-implementing linear algebra operations in Cython (which will likely be both incorrect _and_ slower).

### Existing Cython BLAS wrappers
Correspondingly, there are several ways of doing so.

1. [CythonGSL](https://github.com/twiecki/CythonGSL) provides Cython wrappers for the GNU Scientific Library.
2. [tokyo](https://github.com/tokyo/tokyo) wraps a lot of BLAS routines in Cython functions. 
3. This [StackOverflow](http://stackoverflow.com/questions/16114100/calling-dot-products-and-linear-algebra-operations-in-cython) thread suggests a way of calling the BLAS version bundled with SciPy.

If these projects fit your requirements, great! You can read no further. In my code, however, I often find myself needing only one or two BLAS routines that are called in a tight inner loop -- and in these cases I find it preferable to write my own quick wrapper with just these two functions.

### Calling BLAS directly 

Declaring BLAS functions is a straightforward application of the Cython `cdef extern` machinery.

Getting the BLAS level 1 double inner product function is very straightforward:
```python
cdef extern from 'cblas.h':
    double ddot 'cblas_ddot'(int N,
                             double* X, int incX,
                             double* Y, int incY) nogil
```
This gives a function that takes the length the vectors `N`, the pointers to the first element of `X` and `Y`, and their strides `incX` and `incY`.

Calling it is also very easy:
```python
cpdef double run_blas_dot(double[::1] x,
                          double[::1] y,
                          int dim):

    # Get the pointers.
    cdef double* x_ptr = &x[0]
    cdef double* y_ptr = &y[0]

    return ddot(dim, x_ptr, 1, y_ptr, 1)
```

Declaring level 2 and level 3 functions is a little bit trickier as we need to take care of the various flags passed into the routines. Taking `DGEMV` (double matrix-vector product) as an example, we need:
```python
cdef extern from 'cblas.h':
    ctypedef enum CBLAS_ORDER:
        CblasRowMajor
        CblasColMajor
    ctypedef enum CBLAS_TRANSPOSE:
        CblasNoTrans
        CblasTrans
        CblasConjTrans
    void dgemv 'cblas_dgemv'(CBLAS_ORDER order,
                             CBLAS_TRANSPOSE transpose,
                             int M, int N,
                             double alpha, double* A, int lda,
                             double* X, int incX,
                             double beta, double* Y, int incY) nogil
```
The first two `ctypedef`s give us the flags governing the matrix-vector product operation:

- `CBLAS_ORDER` determines whether the matrix `A` uses row-major or column-major storage (C and Fortran arrays in NumPy parlance), and
- `CBLAS_TRANSPOSE` determines whether the matrix `A` should be transposed for the multiplication.

The final lines gives us the actual function signature. To call it, we need:

- the two parameters above,
- the dimensions of the `A` matrix, `M` by `N`,
- the scaling constants `alpha` and `beta`,
- the pointers to the `A` matrix and `X` and `Y` vector (where the `Y` vector stores the result), and
- the strides of the `X` and `Y` arrays.

To call it:
```python
cpdef run_blas_dgemv(double[:, ::1] A,
                     double[::1] x,
                     double[::1] y,
                     int M,
                     int N,
                     double alpha,
                     double beta):

    cdef double* A_ptr = &A[0, 0]
    cdef double* x_ptr = &x[0]
    cdef double* y_ptr = &y[0]

    dgemv(CblasRowMajor,
          CblasNoTrans,
          M,
          N,
          alpha,
          A_ptr,
          N,
          x_ptr,
          1,
          beta,
          y_ptr,
          1)
```
And that's it: good enough for quick and dirty projects.

### Real-world examples

For some good examples of using Cython BLAS bindings in anger, [sklearn](https://github.com/scikit-learn/scikit-learn/blob/master/sklearn/linear_model/cd_fast.pyx) uses the BLAS headers approach;
[statsmodels](https://github.com/statsmodels/statsmodels/blob/master/statsmodels/tsa/kalmanf/kalman_loglike.pyx) and [gensim](https://github.com/piskvorky/gensim/blob/master/gensim/models/word2vec_inner.pyx) extract function pointers out of `scipy`. 
