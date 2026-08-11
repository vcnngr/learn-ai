# Proposte — revisore C

## Un gate di coerenza numerica automatico

**Tipo:** altro
**Moduli toccati:** tutti (processo, non contenuto)
**Problema che risolve:** quasi tutti i findings di questa review sono numeri
nel testo che non corrispondono all'output del lab (M03, M08, M13, M09, M14,
M15, M02). Il pattern è stabile: il lab viene modificato (o rieseguito con un
seme/versione diversa) e il testo non viene aggiornato. Un gate automatico lo
intercetta al commit.
**Proposta:** uno script `tools/verifica_numeri.py` che (1) esegue ogni lab e
salva l'output, (2) estrae dai file HTML i blocchi `<pre><code>` marcati come
output di lab (serve una convenzione, es. `data-lab="lab_03_2"`), (3) segnala
le righe del testo che contengono numeri non presenti nell'output corrente
del lab corrispondente. Non deve essere perfetto: basta che elenchi i
candidati da guardare. Va nella checklist di rilascio, non nel corso.
**Costo:** 3–4 h di scrittura una tantum; zero ore sul lettore. Totale corso
invariato: 144,5 h.
**Perché non è un finding:** il corso attuale non è *sbagliato* nel non averlo;
i singoli numeri sbagliati sono già elencati come findings. Questa è la
contromisura di processo.

## Dichiarare esplicitamente la configurazione LoRA del caso di studio

**Tipo:** riscrittura (piccola)
**Moduli toccati:** M06, M14, M17, M21
**Problema che risolve:** il caso di studio («il tuo LoRA sul 27B») non ha mai
una configurazione dichiarata: M06 e M17 assumono 0,5% di addestrabili, M14
mostra configurazioni fra 0,045% e 0,246%, e il lettore non può riconciliare i
conti (finding M06↔M14). Poiché il caso attraversa tutto il corso, la sua
configurazione dovrebbe essere un dato, non un'assunzione implicita.
**Proposta:** aggiungere al profilo (`assets/profilo.js`) una sezione
`caso.lora = { r: 32, alpha: 64, target_modules: "tutti e sette" }` scelta in
modo da produrre esattamente lo 0,5% usato in M06/M17 (r≈33 su tutti e sette:
verificato, 0.498%), e una riga in M06: «il LoRA del caso di studio ha
target tutti e sette i moduli a r=32: è da lì che viene lo 0,5%». M14 guadagna
una riga che collega la sua tabella al caso.
**Costo:** 0 h per il lettore (è una riga di contesto), totale invariato.
**Perché non è un finding:** lo è — la contraddizione è nei findings. Questa
proposta è il *modo concreto* di chiuderla, perché tocca più moduli e il
profilo condiviso; la metto qui per valutazione separata del come, non del se.

## Scheda «non verificabile qui» per i due lab GPU

**Tipo:** altro (miglioria al lab)
**Moduli toccati:** M07, M18
**Problema che risolve:** le sezioni GPU di M07 (4–5) e M18 (5) non sono mai
state eseguite, come dichiara il brief. Chiunque riesegua il corso su una
macchina con GPU non ha un riferimento per sapere se l'output che ottiene è
quello atteso o un bug del lab.
**Proposta:** quando i lab verranno eseguiti una prima volta su GPU, salvare
l'output completo in `labs/expected/lab_07_1_gpu.txt` e
`labs/expected/lab_18_1_gpu.txt` e aggiungere al lab un flag `--confronta` che
ristampa i valori attesi a fianco di quelli misurati. Trasforma «non
verificato» in «verificabile da chiunque abbia l'hardware».
**Costo:** 1 h di esecuzione su GPU + 2 h di scrittura del flag; zero ore sul
lettore. Totale invariato.
**Perché non è un finding:** il brief dichiara esplicitamente che le sezioni
GPU non sono verificate e chiede di segnalarlo, non di considerarlo errore.
Il lab si comporta come promesso (fermo pulito senza CUDA — verificato).
