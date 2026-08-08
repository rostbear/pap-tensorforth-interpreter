/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "tokenizer.h"
#include "stack.h"
#include "tensor.h"

// Funzione ausiliaria per leggere il file
char* read_file(const char* filename) {
    FILE* f = fopen(filename, "rb");
    if (f == NULL){
        fprintf(stderr, "Errore: Impossibile aprire il file sorgente\n");
        exit(1);
    }
    fseek(f, 0, SEEK_END);
    long length = ftell(f);
    fseek(f, 0, SEEK_SET);
    char* buffer = malloc(length + 1);
    if (buffer == NULL) {
        fprintf(stderr, "Errore: Allocazione memoria fallita per la lettura del file\n");
        fclose(f);
        exit(1);
    }
    fread(buffer, 1, length, f);
    fclose(f);
    buffer[length] = '\0';

    return buffer;
}

// Variable globale per liberare il sorgente in caso di errore
char* source = NULL;

// Funzione ausiliaria per liberare tutte le risorse ed uscire
void free_all_and_exit(stack_t* stack, int code) {
    if (source != NULL) {
        free(source);
        source = NULL;
    }
    if (stack != NULL) {
        while (stack->top >= 0) {
            stack_item_t item = stack_pop(stack);
            if (item.type == ITEM_TENSOR) {
                tensor_destroy(item.data.tensor);
            } else if (item.type == ITEM_STRING) {
                free(item.data.string);
            }
        }
        stack_destroy(stack);
    }
    exit(code);
}

// Funzione ausiliaria per le operazioni tra due tensori (commutative)
void binary_op(stack_t* stack, tensor_t* (*function)(tensor_t*, tensor_t*)) {
    stack_item_t b = stack_pop(stack);
    stack_item_t a = stack_pop(stack);
    
    if (a.type != ITEM_TENSOR || b.type != ITEM_TENSOR) {
        fprintf(stderr, "Errore: Operazione richiede due tensori\n");
        if (a.type == ITEM_STRING) free(a.data.string);
        if (b.type == ITEM_STRING) free(b.data.string);
        free_all_and_exit(stack, 1);
    }
    
    //Funzione che prende 2 tensori e restituisce un tensore
    tensor_t* result = function(a.data.tensor, b.data.tensor);
    //Inserisce il risultato della funzione nello stack
    stack_push(stack, (stack_item_t){
        .type = ITEM_TENSOR,
        .data.tensor = result
    });
    tensor_destroy(a.data.tensor);
    tensor_destroy(b.data.tensor);
}

// Funzione ausiliaria per le operazioni non commutative tra due tensori
void binary_op_non_commutative(stack_t* stack, tensor_t* (*function)(tensor_t*, tensor_t*)) {
    stack_item_t a = stack_pop(stack); // CIMA dello stack
    stack_item_t b = stack_pop(stack); // SOTTO-CIMA dello stack
    
    if (a.type != ITEM_TENSOR || b.type != ITEM_TENSOR) {
        fprintf(stderr, "Errore: Operazione richiede due tensori\n");
        if (a.type == ITEM_STRING) free(a.data.string);
        if (b.type == ITEM_STRING) free(b.data.string);
        free_all_and_exit(stack, 1);
    }
    
    // Esegue function(cima, sotto-cima) come da specifiche PDF
    tensor_t* result = function(a.data.tensor, b.data.tensor);
    //Inserisce il risultato della funzione nello stack
    stack_push(stack, (stack_item_t){
        .type = ITEM_TENSOR,
        .data.tensor = result
    });
    tensor_destroy(a.data.tensor);
    tensor_destroy(b.data.tensor);
}

// Funzione ausiliaria per le operazioni su un tensore
void unary_op(stack_t* stack, tensor_t* (*function)(tensor_t*)) {
    stack_item_t a = stack_pop(stack);
    
    if (a.type != ITEM_TENSOR) {
        fprintf(stderr, "Errore: Operazione richiede un tensore\n");
        if (a.type == ITEM_STRING) free(a.data.string);
        free_all_and_exit(stack, 1);
    }
    
    //Funzione che prende 1 tensore e restituisce 1 tensore
    tensor_t* result = function(a.data.tensor);
    //Inserisce il risultato della funzione nello stack
    stack_push(stack, (stack_item_t){
        .type = ITEM_TENSOR,
        .data.tensor = result
    });
    tensor_destroy(a.data.tensor);
}


int main(int argc, char* argv[]) {
    if (argc < 2){
        fprintf(stderr, "Errore: Numero di argomenti errato. Esempio di comando: tensorforth [nome file sorgente]\n");
        exit(1);
    }
    source = read_file(argv[1]);
    const char* input_ptr = source;
    stack_t* stack = stack_create(16);
    token_t token;
    while ((token = next_token(&input_ptr)).type != TOKEN_EOF) {
        if (token.type == TOKEN_TENSOR) {
            stack_push(stack, (stack_item_t){
                .type = ITEM_TENSOR,
                .data.tensor = token.tensor_value
            });
        } 
        else if (token.type == TOKEN_STRING) {
            stack_push(stack, (stack_item_t){
                .type = ITEM_STRING,
                .data.string = token.string_value
            });
        } 
        else if (token.type == TOKEN_OPERATOR) {
            switch (token.operator_value) {
                case OP_ADD: binary_op(stack, tensor_add); break;
                case OP_SUB: binary_op_non_commutative(stack, tensor_sub); break;
                case OP_MUL: binary_op(stack, tensor_mul); break;
                case OP_LESS: binary_op_non_commutative(stack, tensor_less); break;
                case OP_MORE: binary_op_non_commutative(stack, tensor_more); break;
                case OP_EQUAL: binary_op(stack, tensor_equal); break;
                case OP_AND: binary_op(stack, tensor_and); break;
                case OP_OR: binary_op(stack, tensor_or); break;
                case OP_MATMUL: binary_op_non_commutative(stack, tensor_matmul); break;
                case OP_DOT: binary_op(stack, tensor_dot); break;
                case OP_CONV2D: binary_op(stack, tensor_conv2d); break;
                case OP_MAX: binary_op(stack, tensor_max); break;
                case OP_MIN: binary_op(stack, tensor_min); break;
                case OP_FILL: binary_op(stack, tensor_fill); break;
                case OP_NOT: unary_op(stack, tensor_not); break;
                case OP_RELU: unary_op(stack, tensor_relu); break;
                case OP_SUM:unary_op(stack, tensor_sum); break;
                case OP_RAVEL: unary_op(stack, tensor_ravel); break;
                case OP_SHAPE: unary_op(stack, tensor_shape); break;
                case OP_RANDOM: unary_op(stack, tensor_rand); break;
                // Caso dove ci sono 3 tensori
                case OP_SELECT: {
                    stack_item_t m = stack_pop(stack);
                    stack_item_t a = stack_pop(stack);
                    stack_item_t b = stack_pop(stack);
                    if (m.type != ITEM_TENSOR || a.type != ITEM_TENSOR || b.type != ITEM_TENSOR) {
                        fprintf(stderr, "Errore: Operazione di selezione richiede tre tensori\n");
                        if (m.type == ITEM_STRING) free(m.data.string);
                        if (a.type == ITEM_STRING) free(a.data.string);
                        if (b.type == ITEM_STRING) free(b.data.string);
                        free_all_and_exit(stack, 1);
                    }
                    tensor_t *result = tensor_select(
                        m.data.tensor,
                        a.data.tensor,
                        b.data.tensor
                    );
                    stack_push(stack, (stack_item_t){
                        .type = ITEM_TENSOR,
                        .data.tensor = result
                    });
                    tensor_destroy(m.data.tensor);
                    tensor_destroy(a.data.tensor);
                    tensor_destroy(b.data.tensor);
                    break;
                }

                case OP_RESHAPE: {
                    stack_item_t shape_item = stack_pop(stack);
                    stack_item_t a_item = stack_pop(stack);
                    if (shape_item.type != ITEM_TENSOR || a_item.type != ITEM_TENSOR) {
                        fprintf(stderr, "Errore: Reshape richiede due tensori\n");
                        if (shape_item.type == ITEM_STRING) free(shape_item.data.string);
                        if (a_item.type == ITEM_STRING) free(a_item.data.string);
                        free_all_and_exit(stack, 1);
                    }
                    int new_ndim = shape_item.data.tensor->shape[0];
                    int new_shape[MAX_DIM] = {0};
                    for (int i = 0; i < new_ndim && i < MAX_DIM; i++) {
                        new_shape[i] = (int)shape_item.data.tensor->data[i];
                    }
                    tensor_t *result = tensor_reshape(
                        a_item.data.tensor,
                        new_shape,
                        new_ndim
                    );
                    stack_push(stack, (stack_item_t){
                        .type = ITEM_TENSOR,
                        .data.tensor = result
                    });
                    tensor_destroy(a_item.data.tensor);
                    tensor_destroy(shape_item.data.tensor);
                    break;
                }
                // Stampa il tensore con distinzione tra stringhe e tensori
                case OP_PRINT: {
                    stack_item_t a = stack_pop(stack);
                    if (a.type == ITEM_TENSOR) {
                        tensor_print(a.data.tensor);
                        tensor_destroy(a.data.tensor);
                    } else if (a.type == ITEM_STRING) {
                        printf("%s\n", a.data.string);
                        free(a.data.string);
                    }
                    break;
                }
                // Stack operations
                case OP_DUP:  stack_dup(stack); break;
                case OP_SWAP: stack_swap(stack); break;
                case OP_OVER: stack_over(stack); break;
                case OP_DROP: stack_drop(stack); break;

                // IO operations
                case OP_IMG_READ: {
                    stack_item_t file = stack_pop(stack);
                    if (file.type != ITEM_STRING) {
                        fprintf(stderr, "Errore: Lettura immagine richiede un nome file (stringa)\n");
                        if (file.type == ITEM_TENSOR) tensor_destroy(file.data.tensor);
                        free_all_and_exit(stack, 1);
                    }
                    tensor_t *res = tensor_from_pgm(file.data.string);
                    stack_push(stack, (stack_item_t){
                        .type = ITEM_TENSOR,
                        .data.tensor = res
                    });
                    free(file.data.string);
                    break;
                }
                case OP_IMG_WRITE: {
                    stack_item_t file = stack_pop(stack);
                    stack_item_t tensor = stack_pop(stack);
                    if (file.type != ITEM_STRING || tensor.type != ITEM_TENSOR) {
                        fprintf(stderr, "Errore: Scrittura immagine richiede un nome file (stringa) e un tensore\n");
                        if (file.type == ITEM_TENSOR) tensor_destroy(file.data.tensor);
                        if (file.type == ITEM_STRING) free(file.data.string);
                        if (tensor.type == ITEM_TENSOR) tensor_destroy(tensor.data.tensor);
                        if (tensor.type == ITEM_STRING) free(tensor.data.string);
                        free_all_and_exit(stack, 1);
                    }
                    tensor_to_pgm(tensor.data.tensor, file.data.string);
                    tensor_destroy(tensor.data.tensor);
                    free(file.data.string);
                    break;
                }
                case OP_TENSOR_READ: {
                    stack_item_t file = stack_pop(stack);
                    if (file.type != ITEM_STRING) {
                        fprintf(stderr, "Errore: Lettura tensore richiede un nome file (stringa)\n");
                        if (file.type == ITEM_TENSOR) tensor_destroy(file.data.tensor);
                        free_all_and_exit(stack, 1);
                    }
                    tensor_t *res = tensor_from_file(file.data.string);
                    stack_push(stack, (stack_item_t){
                        .type = ITEM_TENSOR,
                        .data.tensor = res
                    });
                    free(file.data.string);
                    break;
                }
                case OP_TENSOR_WRITE: {
                    stack_item_t file = stack_pop(stack);
                    stack_item_t tensor = stack_pop(stack);
                    if (file.type != ITEM_STRING || tensor.type != ITEM_TENSOR) {
                        fprintf(stderr, "Errore: Scrittura tensore richiede un nome file (stringa) e un tensore\n");
                        if (file.type == ITEM_TENSOR) tensor_destroy(file.data.tensor);
                        if (file.type == ITEM_STRING) free(file.data.string);
                        if (tensor.type == ITEM_TENSOR) tensor_destroy(tensor.data.tensor);
                        if (tensor.type == ITEM_STRING) free(tensor.data.string);
                        free_all_and_exit(stack, 1);
                    }
                    tensor_to_file(tensor.data.tensor, file.data.string);
                    tensor_destroy(tensor.data.tensor);
                    free(file.data.string);
                    break;
                }
                default: {
                    fprintf(stderr, "Errore: Operazione o token non riconosciuto\n");
                    free_all_and_exit(stack, 1);
                }
            }
        }
    }
    free(source);
    // Svuotamento e distruzione dello stack
    while (stack->top >= 0) {
        stack_item_t item = stack_pop(stack);
        if (item.type == ITEM_TENSOR) {
            tensor_destroy(item.data.tensor);
        } else if (item.type == ITEM_STRING) {
            free(item.data.string);
        }
    }
    stack_destroy(stack);

    return 0;
}
