---
layout: post
title: Incremental construction of sparse matrices
excerpt: ""
tags: [scipy, sparse]
modified: 2015-02-22
comments: true
---

Sparse matrices are an indispensable tool -- because only non-zero entries are stored, they store information efficiently and enable (some) fast linear algera operations.

In Python, sparse matrix support is provided by `scipy` in [scipy.sparse](http://docs.scipy.org/doc/scipy/reference/sparse.html). They come in a number of flavours. Crucially, there are those that use efficient storage and/or support fast linear algebra operations (`csr_matrix`, `csc_matrix`, and `coo_matrix`), and those that enable efficient incremental construction and/or random element access (`lil_matrix`, `dok_matrix`).

A typical use case for me is constructing a sparse matrix incrementally: I may know the shape of the matrix in advance, but do not have all the elements in advance (say, I am reading the matrix from a file element-by-element).

The scipy [docs](http://docs.scipy.org/doc/scipy/reference/sparse.html#usage-information) suggest I use either `dok_matrix` or `lil_matrix` for that, and then convert to a more efficient representation. 

This works very well for small matrices, but for matrices with hundreds of millions of elements this simply does not work: I run out of memory whilst constructing my matrix even though _I know_ that the resulting CSR matrix fits comfortably in RAM. Why is this?

### Delving into scipy.sparse internals

Looking into the [source](https://github.com/scipy/scipy/blob/master/scipy/sparse/lil.py) for `lil_matrix`, we can see that it stores the matrix elements in a numpy array (of `dtype` `object`) of Python lists:

```python
self.shape = (M,N)
self.rows = np.empty((M,), dtype=object)
self.data = np.empty((M,), dtype=object)
for i in range(M):
    self.rows[i] = []
    self.data[i] = []
```

These lists are then populated with column indices and entry values.

The problem with this is that Python lists are incredibly inefficient at storing large numbers of small objects of the same type. In CPython, they are implemented as arrays of pointers to actual list elements. If we would like to store 100 32 bit integers in a Python list on a 64 bit system, CPython would allocate an array of (at least) 100 64 bit pointers, making the list overhead _twice_ the size of the data we actually want to store.

To make matters worse, a CPython 32 bit integer is represented by an instance of [PyObject](https://docs.python.org/2/c-api/structures.html#c.PyObject), which itself imposes additional memory overhead (the reference count of a given object, for example).

It is this overhead that makes using `lil_matrix` (or `dok_matrix`, which uses a Python dictionary) problematic when constructing large matrices.

### The array module to the rescue

What we really want, then, is a list-like object that stores numerical data efficiently. This is precisely what the [array module](https://docs.python.org/2/library/array.html) provides. The `array.array` objects are like lists in that they support appending, but like numpy arrays in that they store their data directly in a typed buffer (and so are similar to a C++ `vector` or a Java `ArrayList`).

What is more, because they support the [buffer protocol](https://jakevdp.github.io/blog/2014/05/05/introduction-to-the-python-buffer-protocol/), it is possible to create a numpy array from an `array.array` _without copying the underlying data_.

This is perfect for implementing an incremental sparse array constructor. The following is a simple example:

```python
import array
import numpy as np
import scipy.sparse as sp


class IncrementalCOOMatrix(object):

    def __init__(self, shape, dtype):

        if dtype is np.int32:
            type_flag = 'i'
        elif dtype is np.int64:
            type_flag = 'l'
        elif dtype is np.float32:
            type_flag = 'f'
        elif dtype is np.float64:
            type_flag = 'd'
        else:
            raise Exception('Dtype not supported.')

        self.dtype = dtype
        self.shape = shape

        self.rows = array.array('i')
        self.cols = array.array('i')
        self.data = array.array(type_flag)

    def append(self, i, j, v):

        m, n = self.shape

        if (i >= m or j >= n):
            raise Exception('Index out of bounds')

        self.rows.append(i)
        self.cols.append(j)
        self.data.append(v)

    def tocoo(self):

        rows = np.frombuffer(self.rows, dtype=np.int32)
        cols = np.frombuffer(self.cols, dtype=np.int32)
        data = np.frombuffer(self.data, dtype=self.dtype)

        return sp.coo_matrix((data, (rows, cols)),
                             shape=self.shape)

    def __len__(self):

        return len(self.data)
```

A quick test to show that it works (and that the data are not copied when converting to a `coo_matrix`):
```python
def test_incremental_coo():

    shape = 10, 10

    dense = np.random.random(shape)
    mat = IncrementalCOOMatrix(shape, np.float64)

    for i in range(shape[0]):
        for j in range(shape[1]):
            mat.append(i, j, dense[i, j])

    coo = mat.tocoo()

    assert np.all(coo.todense() == sp.coo_matrix(dense).todense())
    assert coo.row.base is mat.rows
    assert coo.col.base is mat.cols
    assert coo.data.base is mat.data
```

The same approach applies to incrementally constructing a CSR matrix. Assuming that data come in order a row at a time, it's easy to incrementally grow the three CSR data arrays, and convert them to a `csr_matrix` without copying the underlying memory.

(One caveat here is that `array` overallocates space when it grows. It is quite likely, therefore, that the actual memory usage will be greater than is necessary. Still, this overhead is small relative to the overhead of using an untyped Python container.)
