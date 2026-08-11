# L'AMBIENTE DI RIFERIMENTO DEL CORSO.
#
# Fino all'11 agosto 2026 «ambiente di riferimento» voleva dire il portatile
# su cui il corso è stato scritto. Non era riproducibile da nessuno, e la CI
# l'ha dimostrato: lo STESSO torch 2.2.2 su Linux produce numeri diversi da
# macOS su tutti i lab che addestrano. Stessi semi, BLAS diverso.
#
# È la lezione di M15 avverata sul corso stesso: i semi sono condizione
# necessaria, non sufficiente. La contromisura è quella che M15 insegna —
# dichiarare l'ambiente — portata alle sue conseguenze: l'ambiente diventa
# un artefatto versionato, non una macchina.
#
#   docker build -t learn-ai .
#   docker run --rm -v "$PWD:/w" -w /w learn-ai python3 corso/verifica.py
#
# I numeri pubblicati nelle pagine sono l'output di QUESTO container.

FROM python:3.12-slim-bookworm

# node serve al parity fra assets/conti.js e lab_06_1_memoria.py
RUN apt-get update \
 && apt-get install -y --no-install-recommends nodejs \
 && rm -rf /var/lib/apt/lists/*

# Versioni pinnate, non "latest": è il punto di tutto il file.
# torch da indice CPU — il corso non richiede CUDA fino a M07, e le due
# sezioni che la richiedono si fermano pulite quando manca.
RUN pip install --no-cache-dir \
      --index-url https://download.pytorch.org/whl/cpu \
      torch==2.2.2 \
 && pip install --no-cache-dir \
      numpy==1.26.4 \
      safetensors==0.8.0

# UN SOLO THREAD, ed e' la parte meno ovvia di questo file.
#
# La CI ha scoperto che lo stesso container su una macchina a 2 core e su
# una a 16 produce numeri diversi: il numero di thread cambia l'ORDINE
# delle riduzioni in BLAS, e in virgola mobile l'ordine conta. Pinnare
# torch e numpy non basta se non si pinna anche quello.
#
# Con un thread solo la riduzione e' sequenziale e il risultato non
# dipende piu' da quanti core ha la macchina. Costa in velocita' su
# hardware grande; su questi lab, che lavorano su tensori minuscoli,
# la differenza e' trascurabile.
ENV OMP_NUM_THREADS=1 \
    MKL_NUM_THREADS=1 \
    OPENBLAS_NUM_THREADS=1

WORKDIR /corso

# Dichiara sé stesso all'avvio, invece di lasciarlo indovinare.
CMD ["python3", "corso/labs/lab_00_1_ambiente.py"]
