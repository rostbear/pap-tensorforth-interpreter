# Interprete TensorForth (C)

**Gabriel Tavcar**  
**Matricola: SM3201659**

Questo progetto implementa l'interprete per il linguaggio stack-based TensorForth. L'applicazione esegue operazioni su tensori multidimensionali tramite uno stack dedicato, integrando funzionalità per la parallelizzazione e la gestione efficiente della memoria.

## Struttura dell'implementazione
Ho suddiviso il codice in diversi moduli per mantenerlo modulare e pulito:
- **tokenizer**: Analizza il codice sorgente carattere per carattere, estraendo i token (numeri, stringhe, operatori) e saltando spazi bianchi e commenti.
- **stack**: Gestisce lo stack dei dati (che supporta sia tensori che stringhe), implementando le operazioni fondamentali come SWAP, DUP, OVER e DROP.
- **tensor**: Contiene la logica matematica e di gestione dei tensori multidimensionali. Per prevenire memory leak ho implementato un sistema di reference count. Le operazioni matematiche più pesanti (come matmul o conv2d) sono state parallelizzate usando OpenMP. Inoltre, ho integrato la mappatura di memoria (mmap) per caricare i tensori salvati su file senza sovraccaricare la RAM.
- **main**: Rappresenta l'entry point dell'applicazione. Legge il file sorgente, richiama il tokenizer nel ciclo di esecuzione e applica le varie operazioni sullo stack dei dati.

## Compilazione ed Esecuzione
Per compilare il progetto, posizionarsi nella cartella da terminale ed eseguire il comando:
make

Questo comando compilerà i sorgenti generando l'eseguibile 'tensorforth'.

Se si desidera ripulire i file oggetto (.o) e l'eseguibile generato, si può usare il comando:
make clean

Per avviare l'interprete passando un file sorgente in input:
./tensorforth <percorso_file_sorgente>
