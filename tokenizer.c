/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <ctype.h>

#include "tokenizer.h"
#include "tensor.h"

float *parse_float_array(const char **input, int *out_length) {
    (*input)++; // Salta la parentesi iniziale [
    int capacity = 16;
    int count = 0;
    float *arr = malloc(capacity * sizeof(float));
    if (arr == NULL) {
        fprintf(stderr, "Errore: Malloc per il parsing fallita\n");
        exit(1);
    }
    while (**input != '\0' && **input != ']') {
        // Salta tutti gli spazi tra le lettere
        while (isspace(**input)) {
            (*input)++;
        }

        //Fine del tensore
        if (**input == ']'){
            break;
        }
        // Nessuna virgola ammessa
        if (**input == ',') {
            fprintf(stderr, "Errore: Le virgole non sono ammesse\n");
            free(arr);
            exit(1);
        }

        char *endptr; // Puntatore al primo carattere non consumato dalla funzione strtof
        float val = strtof(*input, &endptr); // Converte la stringa in float e indica dove finisce il numero

        // Nessun carattere consumato
        if (endptr == *input) {
            fprintf(stderr, "Errore: Elemento non valido all'interno del tensore\n");
            free(arr);
            exit(1);
        }

        // Realloc se raddoppia la memoria allocata non basta più
        if (count >= capacity) {
            capacity *= 2;
            float *new_arr = realloc(arr, capacity * sizeof(float));
            if (new_arr == NULL) {
                fprintf(stderr, "Errore: Realloc per il parsing fallita\n");
                free(arr);
                exit(1);
            }
            arr = new_arr; //Aggiornamento del puntatore all'array riallocato
        }
        arr[count++] = val; //Aggiunge il valore ottenuto in precedenza da strtof all'array
        *input = endptr; //Sposta il puntatore all'input
    }
    if (**input == ']') {
        (*input)++; // Consuma la parentesi di chiusura altrimenti significa che il tensore non è stato chiuso e quindi va in errore
    } else {
        fprintf(stderr, "Errore: Tensore non chiuso, manca ]\n");
        free(arr);
        exit(1);
    }
    *out_length = count; //Lunghezza del tensore

    return arr;
}

token_t next_token(const char **input) {
    token_t token;
    // Salta tutti gli spazi tra le lettere
    while (isspace(**input)) {
        (*input)++;
    }
    //Fine del file
    if (**input == '\0') {
        token.type = TOKEN_EOF;
        return token;
    }
    //Inizio di un tensore
    if (**input == '[') {
        int len = 0;
        float *arr = parse_float_array(input, &len); //Legge tutti i float all'interno del tensore
        token.type = TOKEN_TENSOR;
        token.tensor_value = tensor_from_array(arr, len); //Crea il tensore a partire dall'array di float
        free(arr);
        if (token.tensor_value == NULL) {
            fprintf(stderr, "Errore: Creazione del tensore fallita durante il parsing\n");
            exit(1);
        }
        
        return token;
    }
    //Inizio di una stringa
    if (**input == '"') {
        (*input)++; // Consuma la virgoletta di apertura
        int capacity = 16;
        int len = 0;
        char *str = malloc(capacity * sizeof(char)); // Vettore che conterrà la stringa
        if (str == NULL) {
            fprintf(stderr, "Errore: Malloc fallita per il parsing della stringa\n");
            exit(1);
        }
        while (**input != '"' && **input != '\0') { // Ciclo che continua finché non si trova una virgoletta o la fine del file
            // Se la memoria allocata non basta, raddoppia la memoria
            if (len + 1 >= capacity) { 
                capacity *= 2;
                char *new_str = realloc(str, capacity * sizeof(char));
                if (new_str == NULL) {
                    fprintf(stderr, "Errore: Realloc fallita durante il parsing della stringa\n");
                    free(str);
                    exit(1);
                }
                str = new_str; // Aggiorna il puntatore all'array riallocato
            }
            str[len++] = **input; // Aggiunge il carattere letto all'array
            (*input)++; // Sposta il puntatore all'input
        }
        // Se la stringa non è chiusa
        if (**input == '\0') {
            fprintf(stderr, "Errore: Stringa non chiusa, manca '\"'\n");
            free(str);
            exit(1);
        }
        (*input)++; // Consuma la virgoletta di chiusura
        str[len] = '\0'; // Aggiunge il terminatore di stringa

        token.type = TOKEN_STRING;
        token.string_value = str;
        
        return token;
    }
    // Inizio di un'operazione
    token.type = TOKEN_OPERATOR;
    char c = **input;
    (*input)++;
    // Legge l'operatore e lo memorizza nel token
    switch (c) {
        case '+': token.operator_value = OP_ADD; break;
        case '-': token.operator_value = OP_SUB; break;
        case '*': token.operator_value = OP_MUL; break;
        case '<': token.operator_value = OP_LESS; break;
        case '>': token.operator_value = OP_MORE; break;
        case '=': token.operator_value = OP_EQUAL; break;
        case '&': token.operator_value = OP_AND; break;
        case '|': token.operator_value = OP_OR; break;
        case '!': token.operator_value = OP_NOT; break;
        case '$': token.operator_value = OP_SELECT; break;
        case '@': token.operator_value = OP_MATMUL; break;
        case '.': token.operator_value = OP_DOT; break;
        case 'c': token.operator_value = OP_CONV2D; break;
        case 'r': token.operator_value = OP_RESHAPE; break;
        case '_': token.operator_value = OP_RAVEL; break;
        case '#': token.operator_value = OP_SHAPE; break;
        case '?': token.operator_value = OP_RANDOM; break;
        case 'R': token.operator_value = OP_RELU; break;
        case 'm': token.operator_value = OP_MIN; break;
        case 'M': token.operator_value = OP_MAX; break;
        case 'S': token.operator_value = OP_SUM; break;
        case 'f': token.operator_value = OP_FILL; break;
        case 'p': token.operator_value = OP_PRINT; break;
        case 'd': token.operator_value = OP_DUP; break;
        case 's': token.operator_value = OP_SWAP; break;
        case 'o': token.operator_value = OP_OVER; break;
        case 'D': token.operator_value = OP_DROP; break;
        case '(': token.operator_value = OP_IMG_READ; break;
        case ')': token.operator_value = OP_IMG_WRITE; break;
        case '{': token.operator_value = OP_TENSOR_READ; break;
        case '}': token.operator_value = OP_TENSOR_WRITE; break;
        default: 
            fprintf(stderr, "Errore: Operatore non riconosciuto\n");
            exit(1); 
    }

    return token;
}
