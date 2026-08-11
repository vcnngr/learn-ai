> **Documento storico — non descrive il corso attuale.**
> È la mappa di progetto con cui il corso è stato impostato, conservata per
> provenienza. I «vincoli fissi» sull'hardware qui sotto **non valgono più**: il
> corso è stato generalizzato ai cinque regimi descritti in `corso/AMBIENTI.md`
> e in M18b, e quel profilo è oggi solo il caso svolto di riferimento.

# Percorso ML — Mappa delle fasi, v2

Versione 2, dopo review Codex (44 finding, 11 BLOCKER). Le correzioni accettate e i punti
contestati sono in fondo al documento.

## Vincoli fissi

Hardware, valido per ogni esercizio:
- 2× NVIDIA H200 NVL (Hopper), 141GB HBM **ciascuna** — non un pool da 282GB, vedi nota sotto
- NVLink tra le due GPU
- AMD EPYC 9555P 64-core, 1TB RAM
- RHEL, storage NVMe locale
- vLLM già in produzione

**Nota memoria, da tenere presente in tutto il percorso:** 141GB è il tetto *per GPU*.
NVLink accelera la comunicazione tra le due, non fonde la memoria in un unico spazio.
Ogni calcolo di footprint si fa **per rank**, e cosa risiede su un rank dipende dalla strategia:
DDP replica tutto su entrambe, ZeRO/FSDP partiziona componenti diversi a stage diversi.
I 1TB di RAM e l'NVMe permettono offload: cambiano la soglia di fattibilità, a costo di velocità.

Problema reale aperto: LoRA su modello denso ~27B, risultati sotto le aspettative. Causa ignota.

---

## Mappa delle fasi

Stima riveduta: **30–38 sessioni da 3–4h (~110–150h)**. La stima precedente (85–100h) era
sottostimata per un discente con PyTorch superficiale.

### F0 — Ponte matematico e PyTorch operativo *(nuova)*
Aggiunta perche F4 e F6 avevano un prerequisito silenzioso.
**Risultato osservabile:** derivare a mano il gradiente di un prodotto matriciale e di
softmax+cross-entropy; usare broadcasting, masking e shape discipline senza esitare;
leggere un grafo di autograd con `.grad_fn`.
Attivata o saltata in base alle risposte alle domande diagnostiche.

### F1 — Anatomia dei pesi
**Risultato osservabile:** dato un checkpoint reale, prevedere conteggio parametri **logici** e
byte **fisici** su disco prima di caricarlo, errore <2%, e spiegare ogni scarto tra i due.
**Contenuti:** tensori, shape, dtype, safetensors (formato di training/storage) vs GGUF (formato
di inference con quantizzazione e layout propri — non un'alternativa equivalente), sharding,
naming (q/k/v/o, gate/up/down, embed, lm_head), tied embeddings.
**Correzione v2:** il naming NON e universalmente decodificabile — varia per repo, QKV fused,
shard di tensor parallel, conversioni. Il risultato atteso e *formulare e verificare un'ipotesi*
triangolando nome + config + shape + modulo, non "sapere a memoria".
**Progetto pratico 1 qui.**

### F2a — Byte, dtype e layout di mixed precision
Split di F2. Serve subito perche F3 richiede byte/parametro.
**Risultato osservabile:** per un setup dato, dire quanti byte occupa ogni parametro in ciascun
ruolo (parametro, gradiente, master copy, momento) e da cosa dipende.
**Contenuti:** distinzione netta tra **storage dtype**, **compute format** e **accumulator dtype**.
TF32 non e un dtype di storage: e una modalita di calcolo Tensor Core su input fp32.

### F3 — Aritmetica della memoria (base)
**Risultato osservabile:** calcolare a mano il footprint **per-rank** di un training dato, poi
verificarlo, con scarto <15%, e saper dire quale termine e stato sbagliato.
**Contenuti:** pesi + gradienti + master weights + stati optimizer + attivazioni.
**Correzione v2:** la formula teorica non basta. Vanno distinti: memoria teorica, peak
dell'allocator per-rank (`torch.cuda.max_memory_allocated()`), `reserved` (`memory_reserved()`),
memoria vista dal driver (`nvidia-smi`). Il primo non misura tutto: workspace temporanei,
frammentazione, bucket di comunicazione, logits e loss buffer, allocazioni fuori dal caching
allocator. Punto di misura e warm-up vanno fissati prima.
**KV cache:** trattata qui come **caso di contrasto**, non come termine della formula. Nel training
teacher-forced su sequenza intera non esiste cache autoregressiva; i K/V intermedi sono attivazioni
di autograd. E un concetto diverso. Serve smontarlo presto perche il modello mentale da vLLM porta
a contarla dove non c'e. Torna come termine reale in F12 (rollout RL) e F16 (inference).

### F4 — Forward/backward e autograd
**Risultato osservabile:** prevedere quali valori devono restare disponibili tra forward e backward
e perche; misurare il trade-off reale del gradient checkpointing (memoria risparmiata vs secondi
per step) sull'hardware.
**Contenuti:** autograd, cosa il backward richiede, gradient checkpointing.
**Correzione v2:** la formulazione corretta e "quale valore deve restare disponibile", non "quale
tensore viene copiato": autograd tipicamente trattiene un riferimento allo storage, non una copia.

### F4b — Protocollo di diagnosi *(nuova, ricorrente da qui in poi)*
Codex: la diagnosi metodica arrivava troppo tardi (era solo F15).
**Risultato osservabile:** ogni run successivo del percorso parte con sanity check su subset
minimo, baseline parity, cattura completa della config, e ipotesi falsificabili scritte prima.
Non e una fase isolata: e una pratica che si applica da qui in avanti.

### F5 — Optimizer, LR, batch effettivo
**Risultato osservabile:** da una loss curve anonima, diagnosticare warmup assente / LR troppo alto
/ clipping mancante / batch effettivo sbagliato, distinguendoli dal segnale osservabile.
**Contenuti:** SGD → Adam → AdamW → 8-bit/paged, LR schedule, warmup, gradient clipping, batch
size effettivo.
**Correzione v2 (BLOCKER):** "gli stati costano 2× i pesi" era sbagliato. Confondeva numero di
tensori con byte. AdamW tiene due momenti (`m`, `v`), quindi 2 *tensori*; il costo in *byte* dipende
dai dtype. Pesi bf16 (2 B/param) con momenti fp32 (8 B/param totali) = i soli momenti pesano **4×**
i pesi. Aggiungere l'eventuale master copy fp32. Si contabilizza per ruolo, dichiarando
l'implementazione.

### F6 — Architettura del transformer
**Risultato osservabile:** scrivere attention+MLP+norm da zero in PyTorch con equivalenza numerica
verificata contro l'implementazione di riferimento, sotto tolleranza dichiarata.
**Contenuti:** MHA/MQA/GQA, RoPE, normalizzazione, MLP.
**Correzione v2 — GQA:** piu query head condividono gruppi di K/V head; MQA e il caso limite con un
solo KV head. Riduce parametri, compute su K/V e soprattutto la KV cache. **Non** riduce la
dimensione di Q. Le shape e il grouping ratio vanno derivati esplicitamente.
**Correzione v2 — RoPE:** e un meccanismo posizionale, non "context extension". Estrapolare oltre la
lunghezza addestrata degrada. Scaling/interpolazione e training/eval aggiuntivo sono interventi
separati e dipendenti dall'implementazione. Il blocco si chiama "RoPE, limiti di estrapolazione,
metodi di context extension".
**Progetto pratico 2 qui**, ma con scope corretto: e un **test di correttezza architetturale** con
tokenizer e dataset forniti. NON soddisfa ancora l'obiettivo "addestrare un modello da zero" —
quello e il capstone F11b.

### F7 — Tokenizer, chat template, loss masking
**Risultato osservabile:** addestrare un BPE su corpus giocattolo; mostrare con `print()` esattamente
quali token contribuiscono alla loss e quali no; riprodurre un template rotto e mostrare la
differenza nei target di loss.
**Contenuti:** BPE, vocabolario, special token, vocabulary extension, chat template, loss masking.
**Correzione v2:** in F7 si dimostra il *meccanismo* del danno (quali token entrano nella loss).
La *quantificazione* del danno sulla qualita richiede F9 e va rimandata: la training loss puo non
mostrare affatto un degrado downstream.
**Check anticipato sul problema reale:** qui verifichiamo template e loss masking del tuo LoRA 27B.
Due delle cause piu frequenti, a costo quasi zero.

### F8 — Dati
**Risultato osservabile:** dataset con spec esplicita + validator **spec-driven** che dichiara quali
classi di errore rileva e riporta i casi non decidibili.
**Contenuti:** costruzione dataset, packing, deduplica, contaminazione, quantita vs qualita.
**Correzione v2 (BLOCKER):** un validator non intercetta la contaminazione "da solo". Serve un
corpus di riferimento e una definizione operativa di match (esatto / near-dup / semantico).
Template e masking richiedono una spec autorevole contro cui validare.
**Correzione v2 (BLOCKER):** la distillazione **esce da qui**. Dataset sintetico ≠ distillazione.
Vedi F12b.

### F9 — Valutazione
**Risultato osservabile:** un eval che distingue rumore da segnale, con held-out pulito, baseline, e
regression check sulle capacita generali; saper calcolare un intervallo di confidenza su una
differenza di metrica e dire se e distinguibile dal rumore.
**Contenuti:** perche la training loss dice poco, held-out, eval task-specific, LLM-as-judge e
limiti, catastrophic forgetting.
**Correzione v2 (BLOCKER):** aggiunta la statistica mancante — intervalli di confidenza, paired
evaluation, bootstrap adatto alla metrica, sensibilita, multiple comparisons. Senza, la domanda
"quanti esempi servono" non ha risposta.
**Correzione v2:** "non si puo migliorare cio che non si sa misurare" e un obbligo **metodologico**,
non un vincolo tecnico. Tecnicamente si puo addestrare prima e valutare dopo, se base, dati e
checkpoint sono preservati. Il rischio reale e la selezione retroattiva, il leakage e l'assenza di
baseline. L'ordine resta, con la motivazione corretta.

### F10 — Continued pretraining vs SFT *(nuova)*
Codex: il continued pretraining era assente e la distinzione non veniva mai insegnata.
**Risultato osservabile:** dato un obiettivo di adattamento, scegliere e motivare tra continued
pretraining, SFT, o la combinazione — su formato dati, loss, rischi e criteri.

### F11 — LoRA / QLoRA / DoRA
**Risultato osservabile, ora separato per metodo:**
- LoRA: derivare a priori parametri addestrabili e memoria; dimostrare numericamente che il merge
  equivale a base+adapter; sapere cosa cambia `rank` e cosa cambia `alpha`, e cosa non cambia.
- QLoRA: quantizzazione della base, compute dtype, dequantizzazione nel forward, memoria reale
  misurata (non stimata), comportamento dell'optimizer.
- DoRA: cosa cambia nella **parametrizzazione**, non solo in rank/alpha.
**Correzione v2 (BLOCKER):** il titolo prometteva tre metodi, i contenuti coprivano quasi solo LoRA.
Ogni dettaglio qui dipende da libreria e versione: lo stack va pinnato e verificato, non assunto.

### F11b — Fine-tuning completo riproducibile + capstone da zero
**Risultato osservabile:** (a) run end-to-end su modello piccolo, riproducibile da zero, con eval
prima/dopo e regression check; (b) **pretraining didattico su piccola scala**: causal objective,
token budget, split train/validation, sampling, schedule, checkpoint/resume.
**Correzione v2 (BLOCKER):** l'obiettivo "addestrare un modello da zero" era coperto solo da F6, che
e un test di correttezza, non un training. Distinzione ora esplicita: **small-scale pretraining
didattico = raggiungibile**; **frontier-scale pretraining = fuori portata**.
**Correzione v2:** la riproducibilita richiede un manifest machine-readable locale: base model +
revision, dataset revision e split, tokenizer e template, seed, data order, batch effettivo,
optimizer/scheduler, precision, versioni di libreria e driver, config di inference.
(L'infrastruttura di tracking resta tagliata; il manifest no.)
**Progetto pratico 3 qui.**

### F12 — Preference optimization (offline)
**Risultato osservabile:** eseguire un DPO piccolo e leggere loss **piu** statistiche del segnale di
preferenza **piu** eval held-out **piu** check di regressione.
**Correzione v2:** "leggere le metriche giuste, non la loss" era una premessa sbagliata. La loss DPO
resta utile per ottimizzazione e debug; e insufficiente come misura di qualita, non inutile.
**Contenuti:** DPO, ORPO. Prerequisiti espliciti: log-probability, KL, modello di riferimento.

### F12b — RL online e distillazione
Separata da F12 su indicazione Codex: quattro metodi in un blocco non producevano scelta consapevole.
**Risultato osservabile:** sapere perche GRPO/PPO cambiano il profilo di memoria (generazione e
training concorrenti, KV cache dei rollout, reference/reward/value model) e cosa richiedono che non
hai. Per la distillazione: distinguere sequence-level da logit/feature-level, gestire il mismatch di
tokenizer, valutare il costo di generazione del teacher.
**Dichiarato:** su 2 GPU e **dimostrazione del meccanismo**, non validazione a scala.

### F13 — Parallelismo su 2× H200 NVLink
**Risultato osservabile:** prevedere la memoria **per-rank** sotto DDP vs ZeRO 1/2/3; misurare
correttezza, throughput end-to-end, breakdown dello step e scaling reale 1→2 GPU su un workload
rappresentativo.
**Correzione v2:** "saturare entrambe le H200" e banale con un workload sintetico e non insegna
niente. Il progetto 5 diventa: **misurare** dove va il tempo e dove il collo e NVLink e dove e
altro. Con due soli punti non si estrapola lo scaling di un cluster: la conclusione va qualificata.

### F14 — MoE *(specializzazione opzionale)*
**Risultato osservabile:** ispezionare un checkpoint MoE, distinguere parametri totali da attivi,
misurare utilizzo degli expert e skew del carico, spiegare perche il fine-tuning di un MoE differisce
da quello di un denso.
**Correzione v2:** il routing top-k token-choice e comune ma non universale; capacity, token dropping
e load-balancing loss dipendono dall'architettura, ed esistono approcci senza auxiliary loss classica.
"Router collapse" va definito in modo misurabile (entropia di routing, skew, token droppati).
**Nota onesta:** non serve al tuo problema 27B denso. L'hai chiesta, quindi c'e. Saltabile.

### F15 — Parity training/inference
**Spostata prima della diagnosi.** Codex, finding BLOCKER: il mismatch training/inference e una
delle ipotesi candidate per il tuo LoRA 27B. Gli strumenti per testarla devono precedere la diagnosi.
**Risultato osservabile:** parity test progressivo — base → adapter live → merged → converted →
quantized — con lo stack pinnato, isolando in quale passaggio il comportamento cambia.
**Contenuti:** merge, caricamento adapter, conversione, parity di template e tokenizer,
quantizzazione post-training. Tutto fortemente dipendente da versioni e support matrix.

### F16 — Diagnosi metodica del LoRA 27B (capstone)
**Risultato osservabile:** ipotesi ordinate per rapporto informazione/costo, esperimento minimo che
falsifica ciascuna, criterio di decisione scritto **prima** di eseguire.
**Correzione v2:** "causa identificata o esclusione motivata" non e sempre raggiungibile — artefatti
mancanti, run non riproducibile o cause multiple lo impediscono. Output accettabile: cause eliminate,
ranking aggiornato, evidenza raccolta, prossimo esperimento discriminante identificato.
**Progetto pratico 4 qui.**

---

## Tagli confermati
- Serving/deployment vLLM: tagliato. Eccezione in F15: la parity training/inference e correttezza
  del training, non deployment.
- Infrastruttura di experiment tracking (setup W&B/tensorboard, Slurm, Ray): tagliata. Il **manifest
  di esperimento** resta obbligatorio: e riproducibilita, non tooling.
- Data engineering su scala (Spark, dedup su TB, crawling): tagliato. Dedup e contaminazione restano
  come concetti su scala single-node.
- Kernel CUDA / Triton / implementare FlashAttention: tagliato di default, non prerequisito di nulla.
  Fase aggiuntiva opzionale su richiesta.
- Sicurezza/alignment, RLHF con annotatori umani: fuori, richiede pipeline di annotazione.
- **NVFP4 pratico: tagliato per limite hardware.** Le H200 sono Hopper: FP8 e nativo, NVFP4 no
  (formato Blackwell). Su questo nodo resterebbe emulazione software, che non dimostra ne
  comportamento ne prestazioni reali. Resta come nozione concettuale.
  Verifica: `nvidia-smi --query-gpu=compute_cap --format=csv` (attesa 9.0).

## Livello raggiungibile (riformulato dopo review)
**Si:** riprodurre e adattare metodi noti usando le primitive dei framework, su singolo nodo, con
supervisione occasionale. Prevedere la memoria per-rank prima di lanciare. Progettare un eval che
regge statisticamente. Diagnosticare un fine-tuning fallito per esclusione ordinata invece che per
tentativi. Leggere un paper di metodo e capirlo.

**No:**
- "Implementare un paper qualsiasi": promessa illimitata: un paper puo richiedere matematica, kernel,
  dati o scala non coperti. Fuori scope.
- Pretraining frontier-scale (data mixture, LR schedule e scaling law si formano su run da settimane
  e migliaia di GPU).
- Failure mode che emergono solo su scala e durata: divergenze dopo giorni, loss spike correlati a
  shard specifici, guasti hardware, recovery da checkpoint corrotto.
- Full fine-tuning di modelli grandi **residente in HBM senza offload**. Con offload su 1TB di RAM e
  NVMe la soglia si sposta, a costo di velocita spesso inaccettabile. Non e un tetto duro.
- RL su scala (generazione e training concorrenti su molte GPU).
- Contributo di ricerca.

## Punti in cui non ho seguito Codex, e perche
1. **KV cache in F3.** Codex: spostarla in F12/F16, perche nel training teacher-forced non esiste
   cache autoregressiva. Tecnicamente corretto. Resta comunque in F3 come **caso di contrasto
   dichiarato**: il modello mentale da vLLM porta a contarla dove non c'e, e va smontato prima del
   primo calcolo di memoria, non dopo.
2. **Ridurre lo scope per far tornare le ore.** Rifiutato: la copertura e stata richiesta
   esplicitamente. Alzata la stima oraria (110–150h) invece di tagliare contenuti.
3. **MoE.** Codex: non serve agli obiettivi core. Vero. Resta perche richiesta, marcata opzionale.
