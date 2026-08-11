# Confronto delle proposte — auto-revisione, GPT, Kimi

Quindici proposte in tutto: **9** dall'auto-revisione (scritta prima delle esterne),
**3** da GPT (revisore A, correttezza tecnica), **3** da Kimi (revisore C, avversariale).
Qwen (revisore B, lente pedagogica) non ha ancora consegnato.

Questo documento non decide: mette in fila cosa si sovrappone, cosa è già fatto, cosa
costa, e in che ordine converrebbe. Le decisioni restano da prendere.

**Budget:** **149 h** (era 144,5 h): +0,5 h per il lab DPO di §4, +3 h per il caso di
diagnosi di §5, +1 h per M00b. Tetto praticabile dichiarato ~200 h. Margine: **51 h**.

---

## 1. Dove i tre revisori convergono

Tre proposte indipendenti descrivono lo **stesso** intervento. È il segnale più forte
del confronto, perché nessuno dei tre ha visto le proposte degli altri.

### Il gate di coerenza numerica lab → HTML

| Chi | Come lo chiama | Cosa aggiunge di suo |
|---|---|---|
| auto-revisione | «Verifica automatica di coerenza fra pagine e lab» | proporlo come lab di M15, dove il tema è la riproducibilità (+0,5 h) |
| GPT | «Controllo automatico lab → HTML» | uguaglianza esatta per i deterministici, **tolleranze dichiarate** per i float dipendenti dal backend; le sezioni casuali devono usare seed espliciti o non pubblicare valori fissi; il fallimento deve stampare pagina, campo, atteso, osservato |
| Kimi | «Un gate di coerenza numerica automatico» | una **convenzione di marcatura** (`data-lab="lab_03_2"`) per legare ogni blocco al lab che lo produce; accetta falsi positivi purché elenchi i candidati |

**Stato: FATTO.** Le tre richieste sono state applicate a `corso/verifica.py`:

1. **tolleranze dichiarate** (GPT) — tre regole esplicite, `ESATTO` / `ARROTONDA` /
   `REL_FP`, con il rapporto che stampa quante volte ha usato ciascuna. Un numero che
   passa solo per `REL_FP` è marcato come «compatibile con l'ambiente dichiarato», non
   come verificato;
2. **marcatura esplicita** `data-lab=` (Kimi) — 89 blocchi su 89, copertura al 100%.
   Un blocco marcato si confronta **solo** col proprio lab. Esiste anche
   `data-lab="nessuno"` per i blocchi che deliberatamente non vengono da un lab;
3. **posizione nel processo** — `corso/RILASCIO.md`, con codice di uscita 1 in caso di
   difetto e una sezione su cosa il gate *non* copre.

**Il delta ha ripagato subito: la vecchia versione dichiarava zero problemi mentre
sei numeri pubblicati erano sbagliati.** Li nascondeva per tre difetti, tutti chiusi:

| Difetto del gate | Cosa lasciava passare |
|---|---|
| il segno non veniva catturato: `−0,073` cercato come `+0,073` | qualunque errore di segno, in silenzio |
| soglia «numero distintivo» a ≥3 cifre significative | `676`, `4,6`, `321`, `96` — cioè tutti i numeri sbagliati di M13 |
| confronto sull'**unione** dei 24 lab | `676,06` combaciava con un valore di un altro modulo |
| «blocco di output» = almeno due decimali | la tabella token di M13 (`441 / 214 / 64`) e la riga `ppl 694`: zero e un decimale |
| «blocco di codice» riconosciuto da parentesi e `=` | 22 blocchi con tabelle piene di parentesi, esclusi in silenzio dal controllo |

Trovati e corretti così: M13 (`87560.12`, `694.15`, `694` ×3, `214/64`, `6,9×`),
M09 (tre valori RoPE all'ultima cifra), M15 (60/61 byte e i due sha256),
M08 (135 → 132,6 milioni, per coerenza con la config LoRA ora dichiarata).

> Costo: ~3 h di lavoro, **zero ore sul lettore**: questa voce non muove il totale.
> L'idea dell'auto-revisione di presentarlo come lab in M15 (+0,5 h) resta aperta.

---

### I due lab GPU mai eseguiti

| Chi | Proposta |
|---|---|
| GPT | «Matrice di compatibilità per i rami non eseguiti» — conservare output e stato di esecuzione degli stessi lab su `torch 2.2.2`, su una versione recente che attivi `enable_gqa` e `F.rms_norm`, e su almeno un ambiente CUDA per M07/M18. Registrare versione PyTorch/CUDA, GPU, dtype, comando. Nessun benchmark trasferito fra ambienti: il risultato richiesto è **pass/fail dei rami** e coerenza delle formule. |
| Kimi | «Scheda *non verificabile qui*» — alla prima esecuzione su GPU salvare l'output in `labs/expected/lab_07_1_gpu.txt` e `labs/expected/lab_18_1_gpu.txt`, e aggiungere ai lab un flag `--confronta` che ristampa gli attesi a fianco dei misurati. |

Sono la stessa cosa vista da due lati: GPT copre **più assi** (versione di libreria, oltre
alla GPU), Kimi dà il **meccanismo concreto** (file di attesi + flag). Si compongono senza
attrito: la matrice di GPT dice *quali* ambienti registrare, il flag di Kimi dice *come*
renderli utilizzabili da chi riesegue.

**Stato: MECCANISMO FATTO, ESECUZIONE BLOCCATA.** Le due cose erano separabili, e solo la
seconda dipende dall'hardware:

- **`corso/AMBIENTI.md`** — la matrice di GPT. Due ambienti (A: macOS x86 + torch 2.2.2,
  usato per tutto; B: Linux + CUDA + torch recente, **mai eseguito**) per cinque rami. Le
  celle vuote sono dichiarate come *lavoro non fatto*, non come difetti.
- **`--registra` / `--confronta`** — il meccanismo di Kimi, in `lab_07_1` e `lab_18_1`. Gli
  attesi finiscono in `labs/expected/` con in testa una riga che dichiara l'ambiente e una
  **tolleranza dichiarata del 10%**; il confronto stampa misurato, atteso e scarto relativo
  con un verdetto per riga.

Il blocco è **duplicato** nei due lab invece di stare in un modulo condiviso: M00b promette
che i lab sono autonomi e si copiano da soli, e un import fra lab romperebbe quella
promessa. Trenta righe duplicate sono il prezzo, ed è scritto nel codice.

**Come è stato collaudato senza GPU.** `lab_07_1` ha una sezione che misura *su CPU* (la 3,
byte trattenuti dall'attention): il meccanismo è stato agganciato anche a quella, quindi i
tre percorsi — valori entro tolleranza, valore fuori tolleranza, chiave assente — sono
stati provati su dati veri, non simulati. L'unico percorso non collaudato è quello che
richiede numeri veri di GPU, e non poteva esserlo.

> Costo speso: ~2 h, zero ore sul lettore. **Resta 1 h di esecuzione su una macchina CUDA**,
> che non c'è. Quando ci sarà, il lavoro è un comando: `--registra`, guardare l'output
> *prima* di fidarsene, e aggiungere la riga alla matrice.

---

## 2. Proposte già chiuse da questo giro di correzioni

### Dichiarare la configurazione LoRA del caso di studio (Kimi)

Kimi stesso la classificava a metà: «lo è — la contraddizione è nei findings. Questa è il
*modo concreto* di chiuderla». È stata applicata oggi, con una differenza sul numero:

- Kimi proponeva `r=32/33` scelto per **produrre** lo 0,5% già scritto in M06/M17.
- Nessun `r` intero produce 0,5%: `r=32` dà **0,491%** (132 644 864 par.), `r=33` dà 0,507%.
  L'`r` esatto sarebbe 32,57.
- Scelta applicata: **fissare `r=32` e correggere la pagina a 0,491%**, non scegliere l'`r`
  che salva il numero già scritto. La tabella dei GB statici di M06 non cambia a una
  cifra decimale (54,0 / 0,3 / 1,1 / 0,5 / 55,9), M17 passa da 219,8 a 219,7 GB.

La configurazione ora sta in `assets/profilo.js` sotto `caso.lora`, il conto lo stampa
`lab_14_1_lora.py` sez. 1, e `lab_06_1_memoria.py` ha `--lora-frac` con quel default.

> Costo: 0 h per il lettore. **Fatto.**

---

## 3. Le nove proposte dell'auto-revisione

Nessuna delle due revisioni esterne le contraddice, ma **nessuna delle due le sostiene**.
Vale la pena registrare cosa hanno detto GPT e Kimi sulle domande che le riguardano:

| Domanda dell'indice | GPT | Kimi |
|---|---|---|
| L'ordine dei moduli è difendibile? | **Sì.** M12 prima di M13–M17 crea il prerequisito di misura; M20 prima di M21 dà il primo controllo economico. Non propone riordino. | non pronunciato |
| Le durate sono attendibili? | **Non verificabile** dal corpus: mancano tempi osservati su lettori reali del profilo dichiarato. | non pronunciato |
| Ci sono buchi non dichiarati? | **Nessuno dimostrabile.** Esiste un buco di *verifica* (rami CUDA e versioni recenti mai eseguiti), non di contenuto. | non pronunciato |

Questo è il punto più importante del confronto, e va detto chiaro: **il buco «Generazione
e decoding», che l'auto-revisione indicava come la sua proposta singola più forte, non è
stato confermato da nessuno dei due revisori esterni.** GPT ha guardato apposta e ha detto
di non trovare buchi di contenuto dimostrabili. Non è una smentita — GPT leggeva con lente
di correttezza tecnica, non pedagogica — ma è l'esatta lente che manca finché Qwen non
consegna. **Decidere quel modulo prima di Qwen significa decidere senza il revisore che
avrebbe potuto confermarlo o smontarlo.**

Le nove, con il costo dichiarato:

| # | Proposta | Tipo | Costo | Sostegno esterno |
|---|---|---|---|---|
| ~~1~~ | ~~M00b — onboarding~~ **fatto** (§6) | nuovo modulo | +1 h | nessuno, ma coerente col fatto che il corso è appena stato aperto a più persone |
| 2 | M19 (MoE) spostato in appendice | riordino | 0 h | GPT dice esplicitamente di non proporre riordini |
| 3 | M09 spezzato in M09a/M09b | riordino | 0 h | nessuno; GPT non ritiene verificabili le durate |
| 4 | **Generazione e decoding** | nuovo modulo | **+6 h** | **nessuno** — vedi sopra |
| ~~5~~ | ~~M21 — caso di diagnosi svolto~~ **fatto** (§5) | lab | +3 h | il difetto sottostante è nei findings di entrambi |
| ~~6~~ | ~~Percorso ridotto «diagnostico»~~ **fatto** (§6) — 81,5 h, non 75 | metadato | 0 h | nessuno |
| ~~7~~ | ~~Verifica automatica lab↔pagine~~ **fatto** (§1) | infrastruttura | 0 h sul lettore | **GPT + Kimi** |
| 8 | Registro degli errori dell'autore come appendice | appendice | +2 h | nessuno |
| 9 | Modulo sui costi economici (ore-GPU, denaro, calendario) | nuovo modulo | +4 h | nessuno |
| | | **fatte 1, 5, 6, 7** | **+4 h → 149 h** | |
| | | **restano 2, 3, 4, 8, 9** | **+12 h → 161 h** | sospese in attesa di Qwen |

*(Nel documento originale l'apertura dichiarava +34 h e il conto per colonna dava +16,5 h.
La discrepanza era lasciata visibile di proposito. Qui vale la colonna: +16,5 h.)*

---

## 4. La proposta di GPT che nessun altro ha fatto

### Lab DPO in due parti: gradiente diretto vs aggiornamento parametrico

**Moduli toccati:** M16 · **Costo: +0,5 h**

Esercizio in due tempi: (a) calcolare con autograd la loss DPO su **due log-probabilità
scalari indipendenti** e stampare i segni delle derivate; (b) eseguire un update sul
modello vero, a parametri condivisi, e rimisurare entrambe. Chiudere confrontando
«direzione locale della loss» e «esito dopo la proiezione nello spazio dei parametri».

**Stato: FATTO.** Sezione 4b di `lab_16_1_dpo.py`, più la sezione corrispondente in M16.

E ha prodotto un risultato che nessuno dei due revisori aveva previsto, e che rende
l'esercizio migliore di come era stato proposto. La **parte A** conferma i segni attesi
(`dL/dlogp(preferita) = −0,050`, cioè spinta verso l'alto, con peso `σ(−z) = 0,50` a
margine zero). La **parte B**, un solo passo sul modello vero, dà `+0,0004` sulla
preferita: **l'aggiornamento segue il gradiente.** Sono i 300 passi a invertirlo, fino a
`−0,1823`.

Quindi la lezione non è «il gradiente dice una cosa e l'update ne fa un'altra», che era
l'ipotesi implicita nella proposta di GPT. È più precisa: il segnale locale è corretto *e
all'inizio viene pure seguito*; perde dopo, perché il peso `σ(−z)` si spegne mentre
l'ottimizzazione continua e i parametri condivisi trascinano la preferita. Su un passo la
misura conferma la formula, su trecento la smentisce — ed è per questo che serviva
eseguirlo invece di ragionarci.

> Costo effettivo: +0,5 h su M16, da 6 a 6,5 h. **Nuovo totale: 145 h.**

**Effetto collaterale:** i tre blocchi nuovi hanno rivelato che il gate scartava come
«codice sorgente» qualunque blocco con parentesi — 22 blocchi in tutto, mai controllati.
Corretto; la copertura è passata da 63/63 a 85/85.

---

## 5. M21 — il caso di diagnosi svolto (auto-revisione)

**Stato: FATTO.** `labs/lab_21_2_caso.py` + una sezione in M21. M21 passa da 10 a 13 h,
il corso da 145 a **148 h**.

Sessanta servizi con una porta ciascuno; un modello che risponde bene (88,3% esatto); un
fine-tuning aggiuntivo che lo porta a 60,0%, con la maschera della loss sfasata di una
posizione. Il protocollo di 21.1 viene eseguito: passo zero con IC appaiato
(`[-0,417, -0,167]`, esclude lo zero → c'è un fenomeno), inventario degli artefatti,
quattro ipotesi in ordine di costo, correzione di **una sola variabile**, verifica che il
sintomo sparisca (100,0%, IC `[+0,283, +0,517]`). Deterministico, ~23 s su CPU.

**Due cose sono emerse scrivendolo, e sono nel lab:**

1. **Il primo impianto era incoerente.** Con un modello di partenza a 0% il «sintomo»
   risultava *positivo* e il passo zero concludeva «fermati» — mentre il lab proseguiva
   lo stesso. Rifatto: il caso ora è un fine-tuning che *peggiora* un modello che già
   funzionava, che è poi il caso di studio del corso.
2. **Le ipotesi 1 e 3 sono controlli che non possono fallire.** Il modellino non ha
   dropout, quindi `train()` ed `eval()` coincidono per costruzione; e i due testi del
   template escono dalla stessa costante. È il difetto tipo 2 del brief — «procedura che
   non può produrre la risposta che le si chiede» — e sarebbe stato facile spacciarlo per
   verifica riuscita. Il lab lo dichiara sotto ciascuna delle due, e spiega cosa le rende
   informative su un modello vero.

## 6. M00b e il percorso ridotto (auto-revisione)

**Stato: FATTI entrambi.**

**M00b — «Mettere in piedi l'ambiente»** (+1 h, corso a 149 h). Nuovo modulo fra M00 e M01,
con `labs/lab_00_1_ambiente.py`: non installa niente, misura. Stampa interprete, pacchetti,
un controllo di autograd, acceleratori, le API che dipendono dalla versione, e un verdetto
su *quali sezioni del corso questa macchina non potrà eseguire*. Copre anche i tre modi in
cui un lab non gira — modulo assente, `AttributeError` da versione, `no matching
distribution` — e la distinzione fra cosa `profilo.js` fa e cosa non fa.

**Percorso ridotto** (0 h). Tredici moduli marcati `nucleo` in `assets/nav.js`, con badge
nella mappa e due schede in `index.html`.

> **Correzione al numero della proposta.** L'auto-revisione stimava «circa 75 h». Il totale
> vero, calcolato sulle durate attuali, è **81,5 h** — anche perché nel frattempo M21 è
> passato da 10 a 13 h e si è aggiunto M00b. Le due cifre sono ora **calcolate da
> `nav.js`**, non scritte in pagina: se una durata cambia, cambiano con lei. Scriverle a
> mano sarebbe stato il difetto tipo 3 del brief, introdotto di fresco.

---

## 7. Ordine suggerito

Criterio: prima ciò che è già sostenuto da più di un revisore o già quasi fatto, poi ciò
che costa poco e chiude finding esistenti, per ultimo ciò che aggiunge ore senza sostegno
esterno.

| | Cosa | Costo | Perché qui |
|---|---|---|---|
| ~~1~~ | ~~Delta su `verifica.py`~~ — **fatto**, vedi §1 | ~3 h lavoro, 0 h lettore | tre revisori su tre; ha scoperto sei numeri sbagliati |
| ~~2~~ | ~~Lab DPO in due parti (GPT)~~ — **fatto**, vedi §4 | +0,5 h → 145 h | ha corretto anche l'ipotesi con cui era stato proposto |
| ~3~ | Matrice + `--confronta` (GPT + Kimi) — **meccanismo fatto** (§1), resta l'esecuzione | ~2 h spese, 1 h su GPU | due revisori su due; l'unica voce ancora aperta, e solo per l'hardware |
| ~~4~~ | ~~M21 — caso di diagnosi svolto~~ — **fatto**, vedi §5 | +3 h → 148 h | il difetto sottostante era nei findings, non solo nelle proposte |
| ~~5~~ | ~~M00b onboarding~~ — **fatto**, vedi §6 | +1 h → 149 h | il corso è stato appena distribuito; costo basso |
| ~~6~~ | ~~Percorso ridotto~~ — **fatto**, vedi §6. Non 75 h: **81,5 h** misurate | 0 h | costo nullo, ma è una scelta editoriale, non un difetto |
| — | **Generazione e decoding, costi economici, registro errori, riordini M09/M19** | +12 h | **da rivedere dopo Qwen**: sono le proposte che dipendono da una lente pedagogica, ed è l'unica lente che manca |

**Fatte 1, 2, 4, 5 e 6**, e della **3** resta solo l'esecuzione su GPU. Il corso è a
**149 h**. Tutto ciò che non dipendeva dall'hardware è chiuso.

Restano le quattro proposte sospese in attesa di Qwen (+12 h → 161 h), che sono anche le
uniche che nessun revisore esterno sostiene.

---

## 8. Cosa manca a questo confronto

- **Qwen (revisore B, lente pedagogica).** Da rilanciare dopo il 12 agosto 2026 con un
  prompt più leggero: il precedente è stato interrotto dopo 85 minuti per consumo di quota.
  È il revisore le cui domande — ordine dei moduli, prerequisiti silenziosi, durate —
  ricadono esattamente sulle quattro proposte che qui restano senza sostegno.
- **Tempi osservati su lettori reali.** GPT ha dichiarato le durate *non verificabili* dal
  corpus. Finché non esiste almeno un lettore cronometrato, ogni proposta che si giustifica
  con «M09 è troppo lungo» o «149 h scoraggiano» si regge su un'impressione, non su una
  misura — che è precisamente ciò che il corso insegna a non fare. Vale anche per le 81,5 h
  del percorso ridotto: sono la somma di stime, non di tempi osservati.
- **Una macchina con CUDA e un torch recente.** Cinque rami di codice non sono mai stati
  eseguiti (`AMBIENTI.md`). Il meccanismo per chiuderli c'è; manca l'hardware.

---

## Come è finito questo giro

Delle quindici proposte: **sette applicate**, una applicata a metà per vincolo hardware,
quattro sospese in attesa di Qwen, tre superate dai fatti.

Tre cose sono emerse *facendo* il lavoro, e nessuna era nelle proposte:

1. **Il gate dichiarava zero problemi mentre sei numeri pubblicati erano sbagliati.** Cinque
   difetti nello script, tutti in direzione «taci»: segno perso, soglia troppo alta, unione
   dei lab, blocchi non riconosciuti come output, blocchi scartati come codice. Un controllo
   che non fallisce mai va sospettato, non creduto.
2. **La proposta di GPT sul DPO partiva da un'ipotesi sbagliata**, e si è visto solo
   eseguendola: al primo passo la log-probabilità preferita *sale*, seguendo il gradiente.
3. **Il primo impianto del caso di diagnosi era incoerente** — il passo zero concludeva
   «fermati» e il lab proseguiva lo stesso — e due delle sue quattro ipotesi erano controlli
   che non potevano fallire.

Tutte e tre sono lo stesso difetto che il corso insegna a cercare, commesso mentre lo si
insegnava.
