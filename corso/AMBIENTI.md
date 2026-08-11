# Matrice di compatibilità — dove il corso è stato davvero eseguito

Questo file esiste per una ragione sola: **distinguere «non funziona» da «non l'ho mai
provato»**. Sono due affermazioni molto diverse, e confonderle è il difetto che il corso
insegna a non commettere.

Ogni riga è un ambiente. Ogni colonna è un ramo di codice che può comportarsi
diversamente. Le celle vuote non sono un difetto del corso: sono **lavoro non fatto**,
scritto perché si veda.

## Gli ambienti

| # | Ambiente | torch | CUDA | Stato |
|---|---|---|---|---|
| **A** | **container `Dockerfile`** — Linux x86_64, python 3.12 | 2.2.2+cpu | — | **RIFERIMENTO: i numeri delle pagine sono il suo output** |
| B | macOS Intel x86_64 (macchina di scrittura) | 2.2.2 | — | **diverge da A**, vedi sotto |
| C | Linux + CUDA, torch recente | — | — | **mai eseguito** |

### Perché il riferimento è un container e non una macchina

Fino all'11 agosto 2026 «ambiente di riferimento» voleva dire *il portatile su cui il corso
è stato scritto*. Non era riproducibile da nessuno, e la CI l'ha dimostrato al primo colpo:
lo **stesso** `torch 2.2.2` su Linux produce numeri diversi da macOS in **tutti** i lab che
addestrano — 42 valori su 7 pagine.

| | |
|---|---|
| Lab di pura aritmetica — memoria, forme, merge LoRA, parity | **identici** ovunque |
| Lab che addestrano — M03, M08, M12, M13, M19, M21 | **divergono** fra piattaforme |

Stessi semi, stesso codice, BLAS diverso. Alcune divergenze sono all'ultimo bit
(`6620.82` → `6620.85`), altre no (`332.79` → `294.84`): una differenza minima all'inizio,
amplificata su ventimila passi di addestramento.

### E non bastava pinnare le versioni

Il primo container ancora non riproduceva la CI. Il motivo era il **numero di thread**:
il runner GitHub ne ha 2, la macchina di sviluppo 16, e il conteggio dei thread cambia
l'**ordine delle riduzioni** in BLAS — che in virgola mobile non è commutativo.

```
OMP_NUM_THREADS=1  MKL_NUM_THREADS=1  OPENBLAS_NUM_THREADS=1
```

Con un thread solo la riduzione è sequenziale e il risultato smette di dipendere da quanti
core ha la macchina. È nel `Dockerfile`, ed è la riga meno ovvia di tutto il file.

Le divergenze fra macchine sono passate da **42 a 0**. Non «quasi zero»: zero.

**È la lezione di M15 avverata sul corso stesso:** i semi sono condizione necessaria, non
sufficiente. La contromisura è quella che M15 insegna — dichiarare l'ambiente — portata
alle sue conseguenze: l'ambiente diventa un **artefatto versionato**, non una macchina.

```bash
docker build -t learn-ai .
docker run --rm -v "$PWD:/w" -w /w learn-ai python3 corso/verifica.py
```

`torch 2.2.2` resta pinnato perché è l'ultima versione installabile su macOS x86, e serve
che chi scrive il corso possa eseguire i lab. Ma **i numeri pubblicati sono quelli del
container**, e chi lavora su macOS deve aspettarsi che il gate segnali divergenze in locale:
non è un difetto, è l'ambiente B che non è il riferimento.

## I rami, e dove sono stati verificati

| Ramo | Dove | A (container, riferimento) | C (recente, CUDA) |
|---|---|---|---|
| Sezioni CPU di tutti i 27 lab | ovunque | **eseguito**, 27/27, gate verde | non necessario |
| `F.rms_norm` nativo | `lab_09_1`, M09 | **assente**: usato il fallback, `max\|diff\| = 0.000e+00` contro la formula di riferimento | **mai eseguito** |
| `scaled_dot_product_attention(enable_gqa=)` | `lab_09_1`, M09 | **assente**: teste K/V espanse a mano | **mai eseguito** |
| Picco di memoria + gradient checkpointing | `lab_07_1` sez. 4–5, M07 | **si ferma pulito** (verificato) | **mai eseguito** |
| Memoria per rank + scaling DDP | `lab_18_1` sez. 5, M18 | **si ferma pulito** (verificato) | **mai eseguito** |

«Si ferma pulito» significa: il ramo rileva l'assenza di CUDA, stampa un messaggio
esplicito e **non produce numeri stimati**. Quello è verificato. Quello che resta da
verificare è la correttezza numerica dei valori che quel codice produce su hardware vero.

> **Attenzione a cosa vale questa tabella.** Dice dove il codice è stato *eseguito*, non
> che sia corretto ovunque. Un ramo mai eseguito può contenere qualunque cosa: nelle
> sezioni GPU di `lab_18_1`, per esempio, il codice di misura voce-per-voce è stato scritto
> di recente e ha superato solo il controllo di sintassi.

## Come si riempie una cella

I due lab GPU accettano `--registra` e `--confronta`:

```
# la prima volta, su una macchina con GPU:
python3  corso/labs/lab_07_1_backward.py --registra
torchrun --nproc_per_node=2 corso/labs/lab_18_1_parallelismo.py --misura --registra

# da lì in poi, per chiunque:
python3  corso/labs/lab_07_1_backward.py --confronta
```

`--registra` scrive in `corso/labs/expected/<lab>__<sezione>.txt` i valori misurati, con
in testa una riga che dichiara l'ambiente:

```
# ambiente: torch=2.2.2 cuda=- gpu=- piattaforma=Darwin-x86_64
# tolleranza dichiarata: 10% relativo
attn_b1_s128_MB	1.703944
```

`--confronta` li rimette a fianco di quelli appena misurati, con lo scarto relativo e un
verdetto per riga rispetto alla **tolleranza dichiarata del 10%**.

### Cosa aspettarsi da uno scarto

Uno scarto non dice che uno dei due ambienti sia sbagliato: dice che **differiscono**, e
l'intestazione del file dice in cosa. Ma non tutti gli scarti si leggono allo stesso modo:

- **i tempi** cambiano con macchina, driver e carico — uno scarto è normale e non
  significa niente da solo;
- **la memoria** è aritmetica. Uno scarto grande lì è un fatto da capire, non da tollerare;
- **un valore presente da una parte e assente dall'altra** è il segnale più informativo:
  significa che i due ambienti hanno preso rami di codice diversi.

## Stato degli attesi registrati

| File | Ambiente | Registrato |
|---|---|---|
| `lab_07_1_backward__cpu.txt` | B (macOS) | **sì** — serve anche a collaudare il meccanismo su dati veri |
| `lab_07_1_backward__gpu.txt` | C | no |
| `lab_18_1_parallelismo__gpu2.txt` | C | no |

Il file CPU non è un riempitivo: è l'unico modo di sapere che `--registra` e `--confronta`
funzionano davvero. Il percorso positivo, quello fuori tolleranza e quello con una chiave
assente sono stati provati tutti e tre su di lui. L'unico percorso non collaudato è quello
che richiede numeri veri di GPU — e non poteva esserlo.

## Le previsioni dei due lab GPU

Anche M07 e M18 fanno previsioni, non solo dichiarazioni di ignoranza. Sono sulla
*direzione*, mai sulla magnitudine: il quanto dipende da architettura e stack, e
inventarlo sarebbe peggio che tacere.

| Modulo | Previsione | Ti smentisco se |
|---|---|---|
| M07 | il picco con checkpointing è minore, il tempo per passo maggiore | una delle due va nella direzione opposta |
| M07 | il `k` effettivo dal picco totale è > 1 | viene ≤ 1 — sto sbagliando il denominatore |
| M18 | il picco per rank con DDP ≈ conto statico intero | cala aumentando le GPU senza ZeRO attivo |
| M18 | efficienza di scaling sotto il 100% | la superi — c'era un collo di bottiglia a 1 GPU che non ho considerato |
| M18 | somma delle voci < picco misurato | il picco è minore della somma — allora il difetto è in **M06** |

## I regimi di M18b: previsioni in attesa di essere smentite

M18b non fa misure: fa **previsioni**, e dice come falsificarle. Questa è la
tabella dove atterrano le risposte. Ogni cella vuota è una previsione che nessuno
ha ancora messo alla prova.

| Regime | Chi l'ha provato | Prev. 1 capienza | Prev. 2 picco > statico | Prev. 3 vincolo che morde |
|---|---|---|---|---|
| GPU singola / workstation | — | | | |
| Più GPU in un nodo | — | | | |
| Memoria unificata (Spark, Grace) | — | | | |
| Cluster multi-nodo (HPC) | — | | | |
| GPU a noleggio | — | | | |

Il lab genera la scheda da compilare:

```
python3 corso/labs/lab_18b_1_regimi.py --regime <il tuo> --accel N --gb M \
                                       --params <parametri> --metodo <full|lora|qlora>
```

### Cosa vale la pena rimandare indietro

**Una previsione smentita**, prima di tutto: quella corregge il corso. In
particolare la terza, che è la più esposta — «su memoria unificata morde la
banda», «fra nodi l'efficienza crolla rispetto a dentro il nodo» sono
affermazioni strutturali che nessuno qui ha misurato.

**Un numero che non torna** nella prima o nella seconda: se il picco misurato è
*minore* del conto statico, il corso sta contando una voce che non esiste, ed è
un difetto in M06 — non in M18b.

**Un regime che manca.** Cinque non sono tutti i modi in cui si addestra.

### Cosa NON serve rimandare

Una conferma senza numeri. «Ha funzionato» non corregge niente e non si può
verificare: servono i valori misurati e l'ambiente che li ha prodotti, che è poi
lo stesso principio del manifest di M15.

## Se hai una GPU

Sei nella posizione di scoprire un difetto che qui non si poteva vedere. La sequenza utile,
in ordine:

1. `python3 corso/labs/lab_00_1_ambiente.py` — verifica che il tuo ambiente attivi davvero
   i rami che qui mancano;
2. lancia i due lab GPU e **guarda l'output prima di registrarlo**: se un numero non ha
   senso, hai trovato un difetto, non un riferimento;
3. `--registra`, e aggiungi la riga a questa tabella con la versione esatta e il modello di
   GPU.
