# Matrice di compatibilità — dove il corso è stato davvero eseguito

Questo file esiste per una ragione sola: **distinguere «non funziona» da «non l'ho mai
provato»**. Sono due affermazioni molto diverse, e confonderle è il difetto che il corso
insegna a non commettere.

Ogni riga è un ambiente. Ogni colonna è un ramo di codice che può comportarsi
diversamente. Le celle vuote non sono un difetto del corso: sono **lavoro non fatto**,
scritto perché si veda.

## Gli ambienti

| # | Ambiente | torch | CUDA | GPU | Stato |
|---|---|---|---|---|---|
| A | macOS Intel x86_64 (macchina di scrittura) | 2.2.2 | — | — | **usato per tutto il corso** |
| B | Linux + CUDA, torch ≥ 2.5 | — | — | — | **mai eseguito** |

`torch 2.2.2` non è una scelta: su macOS x86 è **l'ultima versione pubblicata**, le build
successive non escono più per quell'architettura. È un vincolo dell'hardware di scrittura,
non una preferenza — e produce tutte le celle vuote della colonna B.

## I rami, e dove sono stati verificati

| Ramo | Dove | A (2.2.2, CPU) | B (recente, CUDA) |
|---|---|---|---|
| Sezioni CPU di tutti i 27 lab | ovunque | **eseguito**, 27/27 | non necessario |
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
| `lab_07_1_backward__cpu.txt` | A | **sì** — serve anche a collaudare il meccanismo su dati veri |
| `lab_07_1_backward__gpu.txt` | B | no |
| `lab_18_1_parallelismo__gpu2.txt` | B | no |

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
