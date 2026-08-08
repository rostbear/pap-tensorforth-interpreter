/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/

#ifndef _STACK_H
#define _STACK_H

#include "tensor.h"
#include <stdlib.h>

typedef enum {
    ITEM_TENSOR,
    ITEM_STRING
} item_type_t;

typedef struct {
    item_type_t type;
    union {
        tensor_t *tensor;
        char *string;
    } data;
} stack_item_t;


typedef struct {
    stack_item_t *items;
    int top;
    int capacity;
} stack_t;


stack_t* stack_create(int initial_capacity);
void stack_push(stack_t *stack, stack_item_t item);
stack_item_t stack_pop(stack_t *stack);
void stack_destroy(stack_t *stack);


void stack_dup(stack_t *stack);
void stack_swap(stack_t *stack);
void stack_over(stack_t *stack);
void stack_drop(stack_t *stack);

#endif
