# TensorForth – Stack-Based Language Interpreter

**A C-based interpreter for a stack-based programming language designed for tensor manipulation, with OpenMP parallelization for computationally intensive operations.**

## Description

This project implements an interpreter for **TensorForth**, a stack-based programming language inspired by Forth.

The interpreter processes programs from left to right using **postfix notation**. Operands, such as tensors and strings, are first pushed onto the stack, while operators consume the required operands and push the resulting values back onto the stack.

The language is designed to operate on **single-precision floating-point tensors**, supporting both one-dimensional (1D) vectors and two-dimensional (2D) matrices. Computationally intensive operations are parallelized using **OpenMP** to improve performance.

### Main Features

* **Tensor Manipulation:** Generation of random tensors and support for element-wise arithmetic, logical, comparison, and selection operations.
* **Advanced Mathematical Operations:** Support for matrix multiplication and 2D convolution using custom kernels, with stride 1 and zero padding.
* **Stack Management:** Operations for duplicating, swapping, and dropping elements at the top of the stack. The interpreter optimizes memory usage by updating references instead of performing unnecessary allocations.
* **Input/Output and Image Processing:** Support for reading and writing images in binary PGM (grayscale) format, with values normalized between 0 and 1.
* **Memory-Mapped File Access:** Support for accessing files directly from disk using `mmap`, avoiding unnecessary memory copies.
* **Parallel Execution:** OpenMP-based parallelization of computationally intensive operations, such as 2D convolution, to improve execution performance.

## Technologies

* **C**
* **OpenMP** – parallel computing
* **GNU Make** – build automation
* **Linux** – development and execution environment

## Compilation and Execution

The project requires a **Linux environment**, such as Ubuntu, and uses a `Makefile` for compilation.

To compile the project, run:

```bash
make
```

To start the interpreter and run a TensorForth program, specify the source file as a command-line argument:

```bash
./tensorforth [source_file]
```

## Project Documentation

The complete language specifications, memory requirements, supported operations, and practical examples, such as image blurring, are available in the following PDF:

[**Download the Project Documentation**](project_C.pdf)
