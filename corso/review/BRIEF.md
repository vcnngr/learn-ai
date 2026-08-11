# Brief di revisione — corso «Capire, addestrare, diagnosticare»

## Cos'è il materiale

Un percorso di apprendimento guidato di **24 moduli, ~157 ore**, che porta da zero
assoluto (nessun prerequisito matematico) alla capacità di diagnosticare con metodo un
fine-tuning che rende meno del previsto.

**Destinatario:** ingegneri di sistema / piattaforma — forti su Linux, rete, storage,
container, serving di modelli; a digiuno di matematica del training e di internals dei pesi.

**Struttura:** `corso/` contiene 25 pagine HTML e `corso/labs/` contiene 27 script Python.
`corso/assets/profilo.js` parametrizza hardware, modello di riferimento e caso di studio.

**Il vincolo autoimposto, ed è il criterio principale di revisione:**
> Nessuna affermazione empirica senza la misura accanto. Ogni numero che compare in una
> pagina deve venire dall'output del lab che lo produce.

## Cosa NON voglio

- Riscritture del materiale
- Suggerimenti stilistici o di tono
- «Potresti anche aggiungere…» senza il formato richiesto (vedi *Proposte*)

---

## I quattro difetti da cercare

Non sono generici: sono gli errori realmente commessi durante la scrittura, trovati da un
gate di revisione automatico che ne ha intercettati circa venticinque. Il pattern è stabile.

### 1. Regola valida in un regime, presentata come universale
*Esempio reale, già corretto:* «la loss iniziale deve valere `log(V)`» — vero per un
modello inizializzato a caso, **falso** per il fine-tuning di un preaddestrato, dove un
valore fra 1 e 3 è normale. Applicato al regime sbagliato produce una diagnosi falsa.

### 2. Procedura che non può produrre la risposta che le si chiede
*Esempio reale, già corretto:* «misura il termine statico sulla GPU per sapere se ci sta».
Se non ci sta, l'errore di memoria arriva **mentre** lo allochi, prima della lettura. Ed è
strutturalmente impossibile che `memory_allocated()` riporti un valore superiore alla
capacità della scheda.

### 3. Numero nel testo che non corrisponde alla misura del lab
*Esempio reale, già corretto:* tabella delle attivazioni in pagina con valori 2,5 volte più
grandi di quelli prodotti dalla formula nel lab.

### 4. Affermazione che il lab dovrebbe dimostrare e non dimostra
*Esempio reale, già corretto:* una sezione che confrontava `"gatto"` e `" gatto"` per
mostrare l'effetto dello spazio iniziale, applicando `.strip()` a entrambe — cioè
confrontando una stringa con sé stessa.

---

## Le tre lenti

Il corpus va letto **per intero** da ciascun revisore. Cambia la lente primaria.

| Revisore | Lente | Domanda guida |
|---|---|---|
| **A** | Correttezza tecnica | Ogni formula è giusta? Ogni numero nel testo corrisponde all'output del lab che lo produce? Esegui i lab e confronta. |
| **B** | Solidità pedagogica e ordine | Un principiante vero può seguire questa sequenza? Dove c'è un prerequisito silenzioso? L'ordine dei moduli regge? |
| **C** | Caccia avversariale | Trova le regole valide in un regime presentate come universali, e le procedure non eseguibili. Parti dai quattro difetti sopra. |

---

## Le cose di cui l'autore è meno sicuro

Consegnate esplicitamente perché sono il punto a più alto rendimento della revisione.

| # | Dove | Cosa verificare |
|---|---|---|
| 1 | M02 | `log(V)` come loss iniziale attesa e i suoi limiti di validità. Corretta due volte: potrebbe restarci ancora un caso non coperto. |
| 2 | M06, M08 | AdamW a 8 byte/parametro e il framing «gli stati costano 4× i pesi in bf16». |
| 3 | M06, M07 | Il fattore `k` nella formula delle attivazioni e la sua dichiarata non-universalità. Il valore 10 è arbitrario. |
| 4 | M09 | La proprietà di RoPE «dipende solo da `m−n`» e il caveat sull'estrapolazione oltre la lunghezza addestrata. |
| 5 | **M16** | **La spiegazione del *perché* `logp(preferita)` scende durante DPO.** Il fenomeno è misurato; la spiegazione potrebbe essere incompleta o parzialmente sbagliata. |
| 6 | **M19** | **La legittimità dell'esperimento.** Il router non collassa spontaneamente e lo sbilanciamento viene *imposto* con un bias. È dichiarato, ma l'esperimento resta debole. |
| 7 | M14 | QLoRA: `0,531` byte/parametro e la stima di ~840 MB di sole scale su 27B. |
| 8 | M20 | «Fondere in bf16 dà un errore 3 000 volte più grande che in fp32» — vero sul singolo strato misurato, ma il framing potrebbe essere fuorviante. |
| 9 | M12 | Il limite superiore dell'1,5% sul tasso di conclusioni invertite, ricavato con la regola del tre da 0 casi su 200. |
| 10 | M21 | L'ordinamento delle nove ipotesi per costo. È un giudizio dell'autore, non una misura. |

---

## Cosa l'autore NON ha potuto verificare

Dichiarato esplicitamente. **Non fidarsi, leggere il codice.**

- **I lab GPU non sono mai stati eseguiti.** La macchina di sviluppo non ha una GPU.
  Riguarda **M07 sezioni 4–5** e **M18 sezione 5**. Il codice è scritto per fermarsi con un
  messaggio esplicito senza CUDA, ma la sua correttezza è **non verificata**.
- **Versione di PyTorch:** i lab sono stati eseguiti su `torch 2.2.2` + `numpy 1.26.4`, il
  massimo disponibile sulla macchina di sviluppo. Due lab contengono percorsi alternativi
  per API che non esistono in quella versione (`enable_gqa`, `F.rms_norm`). Su versioni più
  recenti si attivano rami di codice **mai eseguiti**.
- **Affermazioni dipendenti dalla versione di libreria:** segnalarle, non verificarle.

---

## Formato dei findings

File: `corso/review/<revisore>/findings.md`

```
### <titolo breve>
**File:** percorso · **Sezione:** riferimento
**Severità:** BLOCKER | SHOULD-FIX | NITPICK
**Cosa è sbagliato:** …
**Correzione concreta:** …
```

**Tre vincoli non negoziabili:**

1. **Non inventare** paper, API, nomi di funzioni, iperparametri «consigliati» o benchmark.
2. Dove una cosa **dipende dalla versione** di libreria, dirlo invece di affermare.
3. Se una cosa **non è verificabile** con quello che hai, dirlo invece di indovinare.
   «Non verificabile» è una risposta accettabile e utile.

---

## Documento parallelo: le proposte di miglioramento

Oltre ai findings, ogni revisore **può** produrre proposte di miglioramento al testo o
all'indice. Verranno valutate separatamente e **non applicate in automatico**.

### La regola di separazione

| | **findings** | **proposte** |
|---|---|---|
| Natura | Qualcosa è **sbagliato** | Qualcosa può essere **migliore** |
| Esempi | Affermazione falsa · numero che non corrisponde · procedura non eseguibile · prerequisito silenzioso | Modulo mancante · ordine diverso · spiegazione più chiara · lab aggiuntivo · indice riorganizzato |
| Destino | Va corretto | Valutato dopo |

**Vincolo:** una proposta non può essere un finding travestito. Se una cosa è sbagliata va
nei findings, anche quando la correzione è grossa. Serve a impedire che un errore venga
derubricato a «suggerimento» e resti nel corso.

### Dove

```
corso/review/<revisore>/
    findings.md
    proposte.md            oppure, se sono molte:
    proposte/
        01-indice.md
        02-nuovo-modulo-xyz.md
        03-riscrittura-m09.md
```

### Formato di ogni proposta

```markdown
## <titolo>

**Tipo:** nuovo modulo | riordino | riscrittura | lab aggiuntivo | altro
**Moduli toccati:** M__, M__
**Problema che risolve:** una frase. Se non sai dirlo, non è una proposta.
**Proposta:** concreta — cosa scrivere, dove, con che struttura.
**Costo:** ore stimate di lavoro del lettore, e nuovo totale del corso.
**Perché non è un finding:** il corso attuale non è *sbagliato* su questo punto.
```

### Sul budget di ore

Il corso vale oggi **157 h**. **Allungarlo non è un problema: c'è spazio fino a circa
200 h.** Una proposta non deve quindi togliere spazio a qualcos'altro per essere valida.

Resta però obbligatorio **dichiarare il costo** e il nuovo totale, per due motivi: perché
il lettore possa pianificare, e perché una proposta da 30 h e una da 2 h vanno valutate con
metri diversi anche quando risolvono lo stesso problema.

Se una proposta portasse il totale **oltre le 200 h**, allora sì: indicare cosa si accorcia
o si elimina per far posto.

**«Perché non è un finding»** obbliga a controllare di non aver messo un errore nella
cartella sbagliata.

---

## Tre domande esplicite sull'indice

Sono giudizi dell'autore, non misure. Ogni revisore si pronunci.

1. **L'ordine dei moduli.** M12 (valutazione) sta *prima* di M13–M17 (addestrare), di
   proposito: non si può migliorare ciò che non si sa misurare. M20 (parity) sta *prima* di
   M21 (diagnosi), di proposito: il disallineamento è un'ipotesi candidata e va escluso per
   primo. Sono scelte difendibili o sbagliate?
2. **Le durate.** 157 h totali, con M09 a 12 h e M13 a 4 h. Qualche modulo è sottostimato
   in modo grave? Il tetto praticabile è ~200 h: se un modulo va allungato, dillo.
3. **I buchi.** Sono escluse *per scelta dichiarata*: prompt engineering, RAG, agenti,
   deployment, orchestrazione, monitoring, kernel CUDA/Triton, RLHF con annotatori umani.
   **C'è un buco che invece non è dichiarato?**

---

## Nota finale

L'autore ha già prodotto la propria auto-revisione in `corso/review/claude/`, scritta
**prima** di ricevere queste revisioni, per non limitarsi ad annuire. Leggerla è opzionale:
se preferisci un giudizio non contaminato, ignorala e confrontala dopo.
