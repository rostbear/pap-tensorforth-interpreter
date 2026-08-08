/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/

#ifndef _TENSOR_H
#define _TENSOR_H

#define MAX_DIM 2

#include <stdbool.h>
#include <sys/types.h>

struct tensor{
    int shape[MAX_DIM]; // Support up to MAX_DIM dimensions
    int ndim;
    float * data;
    int ref_count; // For reference counting
    bool on_disk; //Flag to indicate if tensor is memory mapped from disk
    int fd; // File descriptor for the memory mapped file
    off_t offset; // Offset in the file where the tensor data starts (for memory mapped tensors)
};

struct on_disk_tensor {
    int32_t shape[MAX_DIM];
    int32_t ndim;
    off_t data_offset;
};

typedef struct tensor tensor_t;

tensor_t * tensor_create(int * shape, int ndim);
tensor_t * tensor_from_array(float * data, int len);
void tensor_destroy(tensor_t * tensor);

// Math operations
tensor_t * tensor_add(tensor_t * a, tensor_t * b);
tensor_t * tensor_sub(tensor_t * a, tensor_t * b);
tensor_t * tensor_mul(tensor_t * a, tensor_t * b);

// Comparison operations
tensor_t * tensor_less(tensor_t * a, tensor_t * b);
tensor_t * tensor_more(tensor_t * a, tensor_t * b);
tensor_t * tensor_equal(tensor_t * a, tensor_t * b);

// Logical operations
tensor_t * tensor_and(tensor_t * a, tensor_t * b);
tensor_t * tensor_or(tensor_t * a, tensor_t * b);
tensor_t * tensor_not(tensor_t * a);

// Selection operations
tensor_t * tensor_select(tensor_t *m, tensor_t * a, tensor_t * b);

// Tensor operations
tensor_t * tensor_matmul(tensor_t *a, tensor_t *b);
tensor_t * tensor_dot(tensor_t *a, tensor_t *b);
tensor_t * tensor_conv2d(tensor_t *a, tensor_t *kernel);

// Shape operations
tensor_t * tensor_reshape(tensor_t *a, int *s, int new_ndim);
tensor_t * tensor_ravel(tensor_t *a);
tensor_t * tensor_shape(tensor_t * a);

// Random operations
tensor_t * tensor_rand(tensor_t *s);

// Element-wise operations
tensor_t * tensor_max(tensor_t *a, tensor_t *b);
tensor_t * tensor_min(tensor_t *a, tensor_t *b);
tensor_t * tensor_relu(tensor_t *a);

// Reduction operations
tensor_t * tensor_sum(tensor_t *a);

// Filling of tensors
tensor_t * tensor_fill(tensor_t *s, tensor_t *v);

// Utility functions
void tensor_print(tensor_t * a);

// IO operations
tensor_t * tensor_from_file(const char *filename);
void tensor_to_file(tensor_t * tensor, const char *filename);
tensor_t * tensor_from_pgm(const char *filename);
void tensor_to_pgm(tensor_t * tensor, const char *filename);

#endif
