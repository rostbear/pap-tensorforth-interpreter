/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include "stack.h"
#include "tensor.h"

/*
Crea e inizializza una nuova struttura dati a pila (stack) con una capacità iniziale predefinita. Gestisce l'allocazione dinamica della memoria per la 
struttura principale e per l'array degli elementi, terminando l'esecuzione del programma in caso di errore di allocazione.
*/
stack_t* stack_create(int initial_capacity){
    if (initial_capacity < 1){
        fprintf(stderr, "Errore: Capacità iniziale dello stack minore di 1\n");
        exit(1);
    }
    stack_t*stack = malloc(sizeof(stack_t));
    if(stack == NULL){
        fprintf(stderr, "Errore: Impossibile allocare memoria per lo stack\n");
        exit(1);
    }
    stack->items = malloc(initial_capacity*sizeof(stack_item_t));
    if (stack->items == NULL){
        fprintf(stderr, "Errore: Malloc dello stack fallita\n");
        free(stack);
        exit(1);
    }
    stack->capacity = initial_capacity;
    stack->top = -1;

    return stack;
}

/*
Inserisce un nuovo elemento al vertice dello stack. Qualora la capacità massima attualmente allocata venga raggiunta, la funzione provvede a 
riallocare dinamicamente la memoria, raddoppiandone le dimensioni per prevenire errori di overflow durante l'esecuzione.
*/
void stack_push(stack_t *stack, stack_item_t item){
    if(stack->top == stack->capacity-1){
        stack->capacity = stack->capacity * 2; 
        stack->items = realloc(stack->items, stack->capacity * sizeof(stack_item_t));       
        if(stack->items == NULL){
            fprintf(stderr, "Errore: Realloc dello stack fallita\n");
            free(stack);
            exit(1);
        }
    }
    stack->top++; 
    stack->items[stack->top] = item;
}

/*
Estrae e restituisce l'elemento situato al vertice dello stack. Include un controllo rigoroso per prevenire errori di underflow, terminando in 
modo controllato l'esecuzione del programma qualora lo stack risulti vuoto.
*/
stack_item_t stack_pop(stack_t *stack){
    if(stack->top < 0){
        fprintf(stderr, "Errore: Stack vuoto\n");
        stack_destroy(stack);
        exit(1);
    }
    stack_item_t temp = stack->items[stack->top];
    stack->top--;

    return temp;
}

/*
Dealloca in modo sicuro tutta la memoria precedentemente allocata per la struttura dello stack e il relativo array di elementi. Effettua una verifica 
preventiva per assicurarsi che lo stack sia completamente vuoto prima di procedere con la distruzione.
*/
void stack_destroy(stack_t * stack){
    if(stack->top > 0){
        fprintf(stderr, "Errore: Tentativo di distruzione di uno stack non vuoto\n");
        exit(1);
    }
    if (stack->items != NULL){
        free(stack->items);
    }

    free(stack);
}

/*
Implementa l'operazione di duplicazione (comando 'd'). Duplica l'elemento situato al vertice dello stack. Nel caso dei tensori, la funzione si limita 
a incrementare il contatore dei riferimenti (reference count), ottimizzando così le prestazioni e l'utilizzo della memoria senza effettuare copie profonde.
*/
void stack_dup(stack_t *stack){
    stack_item_t temp = stack->items[stack->top];
    if(temp.type == ITEM_TENSOR){
        temp.data.tensor->ref_count++;
    }
    else if(temp.type == ITEM_STRING){
        temp.data.string = strdup(temp.data.string);
        if(temp.data.string == NULL){
            fprintf(stderr, "Errore: Strdup fallita per la duplicazione della stringa sullo stack\n");
            exit(1);
        }
    }

    stack_push(stack, temp);
}

/*
Implementa l'operazione di scambio (comando 's'). Inverte la posizione dei primi due elementi situati al vertice dello stack, previa verifica della 
loro effettiva disponibilità, manipolando unicamente i puntatori all'interno dell'array.
*/
void stack_swap(stack_t *stack){
    if (stack->top < 1){
        fprintf(stderr, "Errore: Elementi insufficienti per l'operazione swap\n");
        exit(1);
    }
    stack_item_t a = stack->items[stack->top];
    stack_item_t b = stack->items[stack->top-1];

    stack->items[stack->top] = b;
    stack->items[stack->top-1] = a;
}

/*
Implementa l'operazione 'over' (comando 'o'). Seleziona il secondo elemento a partire dal vertice dello stack, ne crea un riferimento e lo posiziona in cima.
Anche in questo viene aggiornato il reference count.
*/
void stack_over(stack_t *stack){
    if (stack->top < 1){
        fprintf(stderr, "Errore: Elementi insufficienti per l'operazione over\n");
        exit(1);
    }
    stack_item_t temp = stack->items[stack->top-1];
    if(temp.type == ITEM_TENSOR){
        temp.data.tensor->ref_count++;
    }
    else if(temp.type == ITEM_STRING){
        temp.data.string = strdup(temp.data.string);
        if(temp.data.string == NULL){
            fprintf(stderr, "Errore: Strdup fallita per l'operazione over dello stack\n");
            exit(1);
        }
    }

    stack_push(stack, temp);
}

/*
Implementa l'operazione di rimozione (comando 'D'). Estrae l'elemento al vertice dello stack e gestisce il decremento del reference count nel caso 
si tratti di un tensore. La deallocazione effettiva della memoria viene invocata solo qualora il contatore raggiunga lo zero.
*/
void stack_drop(stack_t *stack){
    stack_item_t temp = stack_pop(stack);
    if(temp.type == ITEM_TENSOR){
        temp.data.tensor->ref_count--;
        if(temp.data.tensor->ref_count == 0){
            tensor_destroy(temp.data.tensor);
        }
    }
}
