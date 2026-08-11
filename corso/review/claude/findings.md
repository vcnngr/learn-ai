# Auto-revisione — findings

Scritta **prima** di ricevere le revisioni esterne, per avere una posizione agli atti e non
limitarsi ad annuire. Sono i punti che so essere deboli, non quelli che spero non lo siano.

---

### I lab GPU non sono mai stati eseguiti

**File:** `labs/lab_07_1_backward.py` sez. 4–5 · `labs/lab_18_1_parallelismo.py` sez. 5
**Severità:** BLOCKER

**Cosa è sbagliato:** il codice è scritto, dichiara di fermarsi senza CUDA — e lo fa — ma il
ramo che si attiva *con* CUDA non è mai stato eseguito. Può contenere errori di sintassi in
percorsi condizionali, API sbagliate, o produrre numeri privi di senso.

Il corso dichiara «nessuna affermazione empirica senza la misura accanto». Su questi due lab
la promessa non è mantenuta: le misure sono *promesse*, non eseguite.

**Correzione concreta:** eseguirli su hardware con GPU prima di considerare il corso
completo. Fino ad allora, aggiungere in cima a M07 e M18 un avviso esplicito che quelle
sezioni sono codice non verificato. Oggi l'avviso c'è nel brief di revisione ma **non nelle
pagine del corso**, che è dove serve al lettore.

---

### Il lab MoE non dimostra il fenomeno che il modulo esiste per insegnare

**File:** `labs/lab_19_1_moe.py` sez. 3 · `m19-moe.html`
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** il collasso del router **non si verifica spontaneamente**. Lo impongo
con un bias, e lo dichiaro apertamente — ma resta il lab più debole del corso: il lettore
non vede il meccanismo di retroazione (esperto scelto → migliora → viene scelto di più), che
è la vera ragione per cui il collasso avviene nei modelli reali.

Quello che il lab misura bene è un'altra cosa, emersa per caso: il costo bidirezionale del
termine di bilanciamento. Utile, ma non è quello che il titolo del modulo promette.

**Correzione concreta:** o si costruisce un compito con struttura latente a blocchi, dove la
specializzazione è possibile e la retroazione può innescarsi — e allora il collasso va
cercato davvero — oppure si cambia il titolo della sezione in «il costo del bilanciamento»
e si sposta il collasso a una descrizione dichiaratamente non dimostrata.

Non ho tentato la prima strada per mancanza di tempo, ed è una scorciatoia che ho preso
consapevolmente.

---

### M17 non ha nessun addestramento eseguibile

**File:** `labs/lab_17_1_rl_distill.py` · `m17-rl.html`
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** è l'unico modulo della Parte V in cui il lettore non addestra niente.
Il lab è aritmetica dei costi più simulazione dell'errore del baseline. Corretto e utile, ma
il corso è costruito sul principio «ogni concetto finisce in un `print()` di qualcosa che
hai fatto girare», e qui il principio non si applica.

Un lettore che arriva a M17 non ha mai visto un ciclo di RL, nemmeno giocattolo.

**Correzione concreta:** aggiungere un GRPO minimo su un compito verificabile
automaticamente — per esempio generare una sequenza di caratteri che soddisfi una regola
controllabile con una funzione. Costo stimato: +2 h di lavoro del lettore.

---

### Il fattore `k` delle attivazioni è un numero inventato

**File:** `labs/lab_06_1_memoria.py` sez. 3 · `m06-memoria.html`
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** `k = 10` è dichiarato come stima e non come misura, ed è corretto
dichiararlo. Ma è comunque un numero scelto da me senza appoggio, e il lettore lo userà per
fare previsioni. La tabella delle attivazioni di M06 è quindi l'unica del corso i cui numeri
non provengono da una misura.

**Correzione concreta:** M07 misura il `k` effettivo, ma solo su GPU. Serve un modo di
ricavarlo su CPU con un modello piccolo — misurando la pendenza rispetto al batch con il
contatore di byte che M07 già implementa — così che la formula di M06 sia tarata su qualcosa
di misurato prima di essere usata.

---

### Sovrapposizione non dichiarata fra M03 e M08 sul learning rate

**File:** `m03-addestrare.html` · `m08-optimizer.html`
**Severità:** NITPICK

**Cosa è sbagliato:** M03 tratta la soglia di divergenza e l'asimmetria del rischio; M08
tratta optimizer, warmup, clipping e batch effettivo. Il confine è ragionevole ma non è mai
enunciato, e un lettore che arriva a M08 può chiedersi perché si riparli di learning rate.

**Correzione concreta:** una riga in apertura di M08 che dica cosa è già stato coperto in
M03 e cosa aggiunge questo modulo.

---

### «Da zero assoluto» non è del tutto vero

**File:** `index.html` · `m00-patto.html`
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** il corso dichiara «nessun prerequisito matematico», ed è mantenuto. Ma
dal primo lab presuppone che il lettore sappia leggere ed eseguire Python, capire un
`for`, una funzione, un'importazione. Non è dichiarato da nessuna parte.

Per il destinatario originale è irrilevante — sa scrivere Python. Ma il corso è stato appena
generalizzato per essere distribuito, e un collaboratore che non programma si blocca al
primo lab senza capire perché.

**Correzione concreta:** dichiarare esplicitamente in `m00` il prerequisito Python, con il
livello richiesto (leggere ed eseguire, non scrivere da zero), e indicare cosa fare se
manca.

---

### I blocchi «Prevedi» dei moduli finali sono più deboli di quelli iniziali

**File:** `m15`–`m21`
**Severità:** NITPICK

**Cosa è sbagliato:** nei primi moduli il Prevedi è una domanda a cui il lettore può
rispondere con un numero, e lo scarto è misurabile. Negli ultimi tende a diventare una
domanda concettuale a risposta aperta — più facile da «rispondere» annuendo.

Il corso dichiara la previsione come meccanismo di verifica principale. Verso la fine il
meccanismo si indebolisce proprio dove il materiale è più difficile.

**Correzione concreta:** riscrivere i Prevedi di M15, M17 e M21 in modo che chiedano un
numero, un ordinamento o una scelta fra alternative concrete, non un giudizio.

---

### Il caso di studio è dichiarato ma mai svolto

**File:** `m21-diagnosi.html`
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** M21 fornisce protocollo, scheda e ordine delle ipotesi. Non contiene
**un esempio svolto**: nessuna diagnosi completa dall'inizio alla fine con numeri veri,
ipotesi escluse una per una e verdetto finale.

Il modulo insegna un metodo senza mostrarlo applicato. È l'equivalente di spiegare la
maschera della loss senza mai stampare un tensore di etichette — cosa che il corso non fa
mai altrove.

**Correzione concreta:** aggiungere un caso svolto, anche costruito ad arte: un fine-tuning
giocattolo deliberatamente rotto in un punto noto, e la diagnosi che lo trova seguendo il
protocollo. Costo stimato: +3 h.

---

### Nessuna verifica automatica di coerenza fra pagine e lab

**File:** infrastruttura
**Severità:** SHOULD-FIX

**Cosa è sbagliato:** i numeri nelle pagine HTML sono stati copiati a mano dall'output dei
lab. Durante la scrittura questo ha prodotto **almeno tre disallineamenti reali**, tutti
trovati per caso rileggendo. Non esiste nessun controllo che li intercetti.

Ogni volta che un lab viene modificato, le pagine che ne citano i numeri possono
silenziosamente diventare false. È esattamente la classe di guasto che il corso insegna a
temere, presente nel corso stesso.

**Correzione concreta:** uno script che esegua tutti i lab, estragga i blocchi `<pre><code>`
delle pagine e segnali quelli che non compaiono nell'output di nessun lab. Non risolve tutto
— i numeri in prosa restano fuori — ma copre il caso più frequente.
