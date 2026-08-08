/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/

#ifndef _TOKENIZER_H
#define _TOKENIZER_H

#include "tensor.h"

enum token_type{
    TOKEN_TENSOR,
    TOKEN_OPERATOR,
    TOKEN_STRING,
    TOKEN_EOF
};

enum operator_type{
    /* Math operations */
    OP_ADD, //+
    OP_SUB, //-
    OP_MUL, //*
    /* Comparision operations */
    OP_LESS, //<
    OP_MORE, //>
    OP_EQUAL, //=
    /* Logical operations */
    OP_AND, //&
    OP_OR, //|
    OP_NOT, //!
    /* Selection operations */
    OP_SELECT, //$
    /* Tensor operations */
    OP_MATMUL, //@
    OP_DOT, //.
    OP_CONV2D, //c
    /* Shape operations */
    OP_RESHAPE, //r
    OP_RAVEL, //_
    OP_SHAPE, //#
    /* Random operations */
    OP_RANDOM, //?
    /* Element-wise operations */
    OP_RELU, //R
    OP_MIN, //m
    OP_MAX, //M
    /* Reduction operations */
    OP_SUM, //S
    /* Filling of tensors */
    OP_FILL, //f
    /* Utiliyu functions */
    OP_PRINT, //p
    /* Stack operations */
    OP_DUP, //d
    OP_SWAP, //s
    OP_OVER, //o
    OP_DROP, //D
    /* IO operations */
    OP_IMG_READ, //(
    OP_IMG_WRITE, //)
    OP_TENSOR_READ, //{
    OP_TENSOR_WRITE, //}
};

typedef struct token {
    int type;
    union {
        tensor_t * tensor_value;
        enum operator_type operator_value;
        char * string_value;
    };
} token_t;

token_t next_token(const char **input);

float *parse_float_array(const char **input, int *out_length);

#endif
