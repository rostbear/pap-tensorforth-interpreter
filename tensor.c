/*
Nome: Gabriel
Cognome: Tavcar
Matricola: SM3201659
*/
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <unistd.h>
#include <sys/stat.h>
#include <sys/mman.h>
#include <stdbool.h>
#include <omp.h>

#include "tensor.h"

/*
Alloca dinamicamente e inizializza un nuovo tensore basandosi sulle dimensioni (shape) e sul numero di dimensioni (ndim) fornite in input. 
Esegue la validazione dei parametri, calcola il volume totale, inizializza i dati a zero tramite calloc e imposta il reference count iniziale a uno.
*/
tensor_t * tensor_create(int * shape, int ndim) {
    if (ndim <= 0 || ndim > MAX_DIM) {
        fprintf(stderr, "Errore: Numero di dimensioni non valido\n");
        exit(1); 
    }
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    if (t == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per il nuovo tensore\n");
        exit(1);
    }
    t->ndim = ndim;
    int volume = 1;
    for (int i = 0; i < MAX_DIM; i++) {
        if (i < ndim) {
            t->shape[i] = shape[i];
            volume *= shape[i];
        } else {
        
            t->shape[i] = 0; 
        }
    }
    t->data = (float *)calloc(volume, sizeof(float));
    if (t->data == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per i dati del tensore\n");
        free(t);
        exit(1);
    }
    t->ref_count = 1;
    t->on_disk = 0;
    t->fd = -1;
    t->offset = 0;

    return t;
}


tensor_t * tensor_from_array(float * data, int len) {
    if (data == NULL || len <= 0) {
        fprintf(stderr, "Errore: Dati non validi o lunghezza errata\n");
        exit(1);
    }
    int shape[1] = {len};
    tensor_t *t = tensor_create(shape, 1);
    if (t == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per il nuovo tensore\n");
        exit(1);
    }
    for (int i = 0; i < len; i++) {
        t->data[i] = data[i];
    }

    return t;
}

/*
Gestisce il rilascio in sicurezza delle risorse associate a un tensore. Decrementa il reference count e, al raggiungimento dello zero, libera la memoria occupata. 
La funzione distingue opportunamente tra memoria heap allocata dinamicamente e memoria mappata da file su disco (munmap).
*/
void tensor_destroy(tensor_t * tensor) {
    if (tensor == NULL) {
        return;
    }
    tensor->ref_count--;
    if (tensor->ref_count > 0) {
        return;
    }
    if (tensor->data != NULL) {
        if (!tensor->on_disk) {
            free(tensor->data);
        } else {
            struct stat sb;
            if (fstat(tensor->fd, &sb) == -1) {
                fprintf(stderr, "Errore: Impossibile recuperare le info del file\n");
            } else {
                if (munmap((char *)tensor->data - tensor->offset, sb.st_size) == -1) {
                    fprintf(stderr, "Errore: Impossibile deallocare la memoria allocata con mmap\n");
                }
            }
            if (close(tensor->fd) == -1) {
                fprintf(stderr, "Errore: Impossibile chiudere il file descriptor\n");
            }
        }
    }
    
    free(tensor);
}


//Math operations
/*
Esegue l'addizione elemento per elemento tra due tensori. Verifica preliminarmente la stretta compatibilità dimensionale degli operandi 
e alloca un tensore per il risultato. Sfrutta le direttive OpenMP per parallelizzare l'esecuzione del ciclo, massimizzando le prestazioni.
*/
tensor_t *tensor_add(tensor_t *a, tensor_t *b){
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }

    return result;
}


tensor_t *tensor_sub(tensor_t *a, tensor_t *b){
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }

    return result;
}


tensor_t *tensor_mul(tensor_t *a, tensor_t *b){
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }

    return result;
}


//Comparison operations
tensor_t *tensor_less(tensor_t *a, tensor_t *b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] < b->data[i]){
            result->data[i] = 1.0;
        } 
        else{
            result->data[i] = 0.0;
        }
    }

    return result;
}


tensor_t *tensor_more(tensor_t *a, tensor_t *b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] > b->data[i]){
            result->data[i] = 1.0;
        } 
        else{
            result->data[i] = 0.0;
        }
    }

    return result;
}


tensor_t *tensor_equal(tensor_t *a, tensor_t *b) {
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] == b->data[i]){
            result->data[i] = 1.0;
        } 
        else{
            result->data[i] = 0.0;
        }
    }

    return result;
}


//Logical operations
tensor_t *tensor_and(tensor_t *a, tensor_t *b){
    if(a->ndim != b->ndim){
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for(int i = 0; i < nelements; i++){
        if(a->data[i] == 1.0 && b->data[i] == 1.0){
            result->data[i] = 1.0;
        }
        else{
            result->data[i] = 0.0;
        }
    }

    return result;
}


tensor_t *tensor_or(tensor_t *a, tensor_t *b){
    if(a->ndim != b->ndim){
        fprintf(stderr, "Errore: I due tensori devono avere lo stesso numero di dimensioni\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for(int i = 0; i < nelements; i++){
        if(a->data[i] == 1.0 || b->data[i] == 1.0){
            result->data[i] = 1.0;
        }
        else{
            result->data[i] = 0.0;
        }
    }

    return result;
}


tensor_t *tensor_not(tensor_t *a){
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for(int i = 0; i < nelements; i++){
        if(a->data[i] == 1.0){
            result->data[i] = 0.0;
        }
        else{
            result->data[i] = 1.0;
        }
    }

    return result;
}


// Selection operations
tensor_t *tensor_select(tensor_t *m, tensor_t *a, tensor_t *b){
    if(a->ndim != b->ndim || a->ndim != m->ndim){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per l'operazione di selezione\n");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i] || a->shape[i] != m->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili per l'operazione di selezione\n");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(m->data[i] == 1.0){
            result->data[i] = a->data[i];
        }
        else{
            result->data[i] = b->data[i];
        }
    }
    
    return result;
}


// Tensor operations
/*
Calcola il prodotto matriciale tra due tensori bidimensionali. Verifica la coerenza algebrica delle dimensioni (colonne della prima matrice uguali 
alle righe della seconda). Il calcolo è parallelizzato tramite direttive OpenMP, applicando 'collapse(2)' per ottimizzare i cicli for annidati.
*/
tensor_t *tensor_matmul(tensor_t *a, tensor_t *b){
    if(a->ndim != 2 || b->ndim != 2){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per il prodotto di matrici");
        exit(1);
    }
    if(a->shape[1] != b->shape[0]){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per il prodotto di matrici. Il numero di colonne della prima matrice deve essere uguale al numero di righe della seconda");
        exit(1);
    }
    int shape[2] = {a->shape[0], b->shape[1]};
    tensor_t *result = tensor_create(shape, 2);
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a->shape[0]; i++){
        for (int j = 0; j < b->shape[1]; j++){
            float sum = 0.0;
            for (int k = 0; k < a->shape[1]; k++){
                int index_a = (i * a->shape[1]) + k;
                int index_b = (k * b->shape[1]) + j;
                sum += a->data[index_a] * b->data[index_b];
            }
            
            int index = (i * b->shape[1]) + j;
            result->data[index] = sum;
        }
    }
    
    return result;
}


tensor_t *tensor_dot(tensor_t *a, tensor_t *b){
    if(a->ndim != 1 || b->ndim != 1){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per il prodotto interno");
        exit(1);
    }
    if(a->shape[0] != b->shape[0]){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori, devono avere la stessa lunghezza");
        exit(1);
    }
    int shape[1] = {1};
    tensor_t *result = tensor_create(shape, 1);
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < a->shape[0]; i++){
        sum += a->data[i] * b->data[i];
    }
    result->data[0] = sum;

    return result;
}

/*
Applica un'operazione di convoluzione bidimensionale tra un tensore matrice di input e un tensore kernel. La logica gestisce implicitamente il padding 
di zeri ai bordi calcolando opportunamente i limiti degli indici. L'estrazione dei valori risultanti è ampiamente parallelizzata tramite OpenMP.
*/
tensor_t *tensor_conv2d(tensor_t *a, tensor_t *kernel){
    if(a->ndim != 2 || kernel->ndim != 2){
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per la convoluzione 2D\n");
        exit(1);
    }
    int shape[2] = {a->shape[0], a->shape[1]};
    tensor_t *result = tensor_create(shape, 2);
    int center_x = kernel->shape[1] / 2;
    int center_y = kernel->shape[0] / 2;
    #pragma omp parallel for collapse(2)
    for (int i = 0; i < a->shape[0]; i++){
        for (int j = 0; j < a->shape[1]; j++){
            float sum = 0.0;
            for (int ki = 0; ki < kernel->shape[0]; ki++){
                for (int kj = 0; kj < kernel->shape[1]; kj++){
                    int row = i + ki - center_y;
                    int column = j + kj - center_x;
                    if (row >= 0 && row < a->shape[0] && column >= 0 && column < a->shape[1]){
                        int index_a = (row * a->shape[1]) + column;
                        int index_k = (ki * kernel->shape[1]) + kj;
                        sum += a->data[index_a] * kernel->data[index_k]; 
                    } 
                }
            }
            int index = (i * a->shape[1]) + j;
            result->data[index] = sum;
        }
    }

    return result;
}


// Shape operations
tensor_t *tensor_reshape(tensor_t *a, int *s, int new_ndim) {
    int old_elements = 1;
    for (int i = 0; i < a->ndim; i++) {
        old_elements *= a->shape[i];
    }
    int new_elements = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_elements *= s[i];
    }
    if (old_elements != new_elements) {
        fprintf(stderr, "Errore: reshape incompatibile\n");
        exit(1);
    }
    tensor_t *result = tensor_create(s, new_ndim);
    #pragma omp parallel for
    for (int i = 0; i < new_elements; i++) {
        result->data[i] = a->data[i];
    }

    return result;
}


tensor_t *tensor_ravel(tensor_t *a) {
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++) {
        nelements *= a->shape[i];
    }
    int new_shape[1] = {nelements};
    tensor_t *result = tensor_create(new_shape, 1);
    if (result == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per il tensore\n");
        exit(1);
    }
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++) {
        result->data[i] = a->data[i];
    }

    return result;
}


tensor_t *tensor_shape(tensor_t *a) {
    int new_shape[1] = {a->ndim};
    tensor_t *result = tensor_create(new_shape, 1);
    for (int i = 0; i < a->ndim; i++) {
        result->data[i] = (float)a->shape[i]; 
    }

    return result;
}


// Random operations
tensor_t *tensor_rand(tensor_t * s){
    if(s->ndim != 1){
        fprintf(stderr, "Errore: Il tensore che definisce le dimensioni deve essere 1D.\n");
        exit(1);
    }
    int new_ndim = s->shape[0];
    if (new_ndim <= 0 || new_ndim > MAX_DIM) {
        fprintf(stderr, "Errore: Il Numero di dimensioni è minore di 0 o maggiore del massimo consentito\n");
        exit(1);
    }
    int new_shape[MAX_DIM] = {0};
    int nelements = 1;
    for (int i = 0; i < new_ndim; i++) {
        new_shape[i] = (int)s->data[i];
        nelements *= new_shape[i];
    }
    tensor_t *result = tensor_create(new_shape, new_ndim);
    for (int i = 0; i < nelements; i++){
        result->data[i] = (float)rand() / RAND_MAX;
    }
    
    return result;
}


// Element-wise operations
tensor_t *tensor_max(tensor_t * a, tensor_t * b){
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I tensori hanno un numero di dimensioni diverse");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] > b->data[i]){
            result->data[i] = a->data[i];
        }
        else{
            result->data[i] = b->data[i];
        }
    }

    return result;
}

tensor_t *tensor_min(tensor_t * a, tensor_t * b){
    if (a->ndim != b->ndim) {
        fprintf(stderr, "Errore: I tensori hanno un numero di dimensioni diverse");
        exit(1);
    }
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        if (a->shape[i] != b->shape[i]){
            fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori");
            exit(1);
        }
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] < b->data[i]){
            result->data[i] = a->data[i];
        }
        else{
            result->data[i] = b->data[i];
        }
    }

    return result;
}


tensor_t *tensor_relu(tensor_t * a){
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        nelements *= a->shape[i];
    }
    tensor_t *result = tensor_create(a->shape, a->ndim);
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        if(a->data[i] < 0){
            result->data[i] = 0.0;
        } else {
            result->data[i] = a->data[i];
        }

    }

    return result;
}


// Reduction operations
tensor_t *tensor_sum(tensor_t * a){
    int nelements = 1;
    for (int i = 0; i < a->ndim; i++){
        nelements *= a->shape[i];
    }
    int new_shape[] = {1};
    tensor_t *result = tensor_create(new_shape, 1);
    float sum = 0.0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < nelements; i++){
        sum += a->data[i]; 
    }
    result->data[0] = sum;

    return result;
}


// Filling of tensors
tensor_t *tensor_fill(tensor_t *s, tensor_t *v){
    if (s->ndim != 1 || v->ndim != 1) {
        fprintf(stderr, "Errore: Dimensioni incompatibili dei tensori per l'operazione di fill");
        exit(1);
    }
    int new_ndim = s->shape[0];
    int nelements = 1;
    int new_shape[new_ndim];
    for (int i = 0; i < new_ndim; i++){
        new_shape[i] = (int)s->data[i];
        nelements *= new_shape[i];
    }
    tensor_t *result = tensor_create(new_shape, new_ndim);
    int v_len = v->shape[0];
    #pragma omp parallel for
    for (int i = 0; i < nelements; i++){
        result->data[i] = v->data[i % v_len];
    }

    return result;
}


// Utility functions
void tensor_print(tensor_t *a){
    int nelements = 1;
    printf("Tensor(shape=[ ");
    for (int i = 0; i < a->ndim; i++){
        printf("%d ", a->shape[i]);
        nelements *= a->shape[i];
    }
    printf("], data=[ ");
    for (int i = 0; i < nelements; i++){
        printf("%f ", a->data[i]);
    }
    printf("])\n");
}


// IO operations
/*
Inizializza un tensore a partire da un file binario su disco utilizzando la tecnica di memory mapping (mmap) per evitare il sovraccarico della RAM. 
Legge l'intestazione 'on_disk_tensor' e gestisce correttamente l'offset dei dati, assicurando il vincolo di allineamento a 64 byte.
*/
tensor_t * tensor_from_file(const char *filename) {
    int fd = open(filename, O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "Errore: Impossibile aprire e leggere il file \n");
        exit(1);
    }
    struct stat sb;
    if (fstat(fd, &sb) == -1) {
        fprintf(stderr, "Errore: Impossibile recuperare le informazioni (stat) del file aperto");
        close(fd);
        exit(1);
    }
    void *mapped_ptr = mmap(NULL, sb.st_size, PROT_READ, MAP_PRIVATE, fd, 0);
    if (mapped_ptr == MAP_FAILED) {
        fprintf(stderr, "Errore: Mmap fallita, impossibile mappare il file in memoria");
        close(fd);
        exit(1);
    }
    struct on_disk_tensor *header = (struct on_disk_tensor *)mapped_ptr;
    tensor_t *t = (tensor_t *)malloc(sizeof(tensor_t));
    if (t == NULL) {
        fprintf(stderr, "Errore: Impossibile allocare memoria per la struttura del tensore");
        munmap(mapped_ptr, sb.st_size);
        close(fd);
        exit(1);
    }
    t->ndim = header->ndim;
    for (int i = 0; i < MAX_DIM; i++) {
        t->shape[i] = header->shape[i];
    }
    t->data = (float *)((char *)mapped_ptr + header->data_offset);
    t->ref_count = 1;
    t->on_disk = true; 
    t->fd = fd;
    t->offset = header->data_offset;

    return t;
}

void tensor_to_file(tensor_t *tensor, const char *filename) {
    int fd = open(filename, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        fprintf(stderr, "Errore: Impossibile aprire o creare il file\n");
        exit(1);
    }
    struct on_disk_tensor header;
    header.ndim = tensor->ndim;
    header.data_offset = 64;
    for (int i = 0; i < MAX_DIM; i++) {
        header.shape[i] = tensor->shape[i];
    }
    if (write(fd, &header, sizeof(struct on_disk_tensor)) == -1) {
        fprintf(stderr, "Errore: Impossibile aprire il file %s per la lettura", filename);
        close(fd);
        exit(1);
    }
    lseek(fd, 64, SEEK_SET);
    int nelements = 1;
    for (int i = 0; i < tensor->ndim; i++) {
        nelements *= tensor->shape[i];
    }
    if (write(fd, tensor->data, nelements * sizeof(float)) == -1) {
        fprintf(stderr, "Errore: Impossibile scrivere i dati del tensore nel file");
        close(fd);
        exit(1);
    }

    close(fd);
}

/*
Carica in memoria un'immagine in formato PGM binario (scala di grigi) e la converte in un tensore bidimensionale. Legge l'intestazione P5 per definire 
le dimensioni della matrice ed esegue la normalizzazione lineare dei valori dei singoli pixel, mappandoli dall'intervallo [0, 255] a [0.0, 1.0].
*/
tensor_t * tensor_from_pgm(const char *filename) {
    FILE *f = fopen(filename, "rb");
    if (f == NULL){
        fprintf(stderr, "Errore: Errore nell'apertura del file");
        exit(1);
    }
    char p[3];
    int width, height, max_val;
    fscanf(f, "%s %d %d %d", p, &width, &height, &max_val);
    fgetc(f); // Per saltare il newline
    int shape[2] = {height, width};
    tensor_t *t = tensor_create(shape, 2);
    if (t == NULL) {
        fclose(f);
        exit(1);
    }
    unsigned char pixel;
    int sum = height * width;
    for (int i = 0; i < sum; i++) {
        if (fread(&pixel, 1, 1, f) != 1){
            break;
        }
        t->data[i] = pixel / 255.0;
    }
    fclose(f);
    
    return t;
}


void tensor_to_pgm(tensor_t *tensor, const char *filename) {
    if (tensor->ndim != 2){
        fprintf(stderr, "Errore: I tensori hanno un numero di dimensioni diverse");
        exit(1);
    }
    FILE *f = fopen(filename, "wb");
    if (f == NULL){
        fprintf(stderr, "Errore: Errore nell'apertura del file");
        exit(1);
    }
    int height = tensor->shape[0];
    int width = tensor->shape[1];
    fprintf(f, "P5\n");
    fprintf(f, "%d %d\n", width, height);
    fprintf(f, "255\n");
    for (int i = 0; i < height * width; i++) {
        float val = tensor->data[i];
        if (val < 0.0){
            val = 0.0;
        }
        if (val > 1.0){
            val = 1.0;
        }
        unsigned char pixel = val * 255.0;
        fwrite(&pixel, 1, 1, f);
    }
    
    fclose(f);
}
