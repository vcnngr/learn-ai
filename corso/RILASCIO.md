# Checklist di rilascio

> **Automatizzata.** Questa checklist gira da sola a ogni push, in
> `.github/workflows/verifica.yml`, su due ambienti: quello di riferimento
> (`torch 2.2.2`, dove il gate deve passare) e uno recente (dove esegue per la
> prima volta i rami che `AMBIENTI.md` dichiara mai eseguiti, e dove una
> divergenza è un dato invece che un difetto).
>
> Resta scritta qui perché va capita, non solo eseguita — e perché serve
> lanciarla in locale prima di pushare.

Da eseguire **prima** di pubblicare una modifica ai lab, ai dati o all'ambiente.
Non fa parte del corso: è manutenzione.

## 0. I gate sanno fallire?

```bash
docker run --rm -v "$PWD:/w" -w /w learn-ai python3 corso/tools/prova_i_gate.py --riferimento
```

Rompe una cosa alla volta e verifica che il controllo corrispondente **nomini la
mutazione nel rapporto**, poi ripristina. Non basta guardare il codice di uscita: su
una macchina dove il gate è già rosso a riposo — la CI, dove la CPU non è quella di
riferimento — iniettare un difetto lascia l'uscita a 1 comunque, e la prova passerebbe
senza aver dimostrato niente. Quindi si verificano tre cose insieme: uscita 1, il
rapporto *nomina* la mutazione, e a corso integro *non* la nomina. Va **prima** degli altri: un verde prodotto da un controllo che non
può fallire è peggio di nessun controllo, perché autorizza a non guardare.

Per il gate del **sito** la prova corrispondente è `deploy/prova-la-prova.sh`:
inietta un difetto vero — toglie `Referrer-Policy` — e verifica che il rapporto
nomini **quella** intestazione, non che ne manchi genericamente una. La firma è
la terna di colonne `1 1 0`: le prime due presenti, la terza no. Cercare la
stringa «INTESTAZIONE MANCANTE» non bastava: un gate che avesse accusato il CSP
al posto del `Referrer-Policy` sarebbe passato lo stesso. Vive lì e non in `prova_i_gate.py` perché
ha bisogno di Docker, che nel container dei lab non c'è.

`--riferimento` pretende verde anche il gate numerico. Usalo **solo qui**: quel gate
dipende dalla CPU, e in CI la macchina non è quella che ha generato i numeri.

Non è un'ipotesi: `verifica.py --copertura` faceva `return 0` incondizionato mentre
la CI lo invocava dichiarandolo «controllo DURO». Se aggiungi un gate, aggiungi lì
la prova che sa fallire.

## 0b. Le intestazioni del sito

```bash
sh deploy/prova-intestazioni.sh vcnngr/learn-ai-sito:<tag>
```

Verifica che **ogni tipo di risposta** — pagina, asset, asset mancante, pagina
mancante — porti CSP, `nosniff` e `Referrer-Policy`, e la politica di cache
giusta. Esce 1 se una manca.

Serve perché in nginx un `add_header` dentro un `location` **azzera** quelli
ereditati dal `server`, e ripetere l'elenco a mano ha prodotto lo stesso difetto
quattro volte di fila — tutte e quattro trovate da una review, nessuna da me.
Ora le intestazioni stanno in un unico file incluso, e questo script controlla
che l'inclusione abbia funzionato ovunque.

**Gira anche in CI**, nel job `sito`, che costruisce l'immagine e lancia questo
controllo a ogni push. Resta scritto qui perché va capito, e perché conviene
lanciarlo in locale prima di pushare — ma non dipende più dal ricordarsene.

## 1. Il gate numerico

```bash
docker build -t learn-ai .
docker run --rm -v "$PWD:/w" -w /w learn-ai python3 corso/verifica.py
```

**Nel container, non a mano.** I numeri pubblicati sono l'output dell'ambiente di
riferimento definito dal `Dockerfile`; lanciarlo sul proprio portatile produce
divergenze che non sono difetti del corso — vedi `AMBIENTI.md`.

Esce con **0** se ogni numero pubblicato risale a un output di lab, con **1**
altrimenti. Non pubblicare con uscita 1 senza aver letto ogni riga del rapporto.

Cosa aspettarsi a corso pulito:

```
blocchi di output marcati     :   63 / 63  (100%)
numeri nei blocchi non trovati : 0
numeri in prosa non trovati    : 0
```

La riga della **copertura** conta quanto le altre due. Un blocco non marcato
viene confrontato con l'unione dei 24 lab: una rete così larga che dichiarava
zero problemi mentre M13 pubblicava `676,06` contro le `694,15` del lab.

## 2. Marcare i blocchi nuovi

Ogni `<pre><code>` che contiene output di lab va marcato con il lab che lo
produce, **senza** il nome descrittivo:

```html
<pre data-lab="lab_13_1"><code>...</code></pre>
```

Se il blocco non viene da un lab — esempio didattico, schema inventato,
formato illustrativo — va marcato lo stesso:

```html
<pre data-lab="nessuno"><code>...</code></pre>
```

«Non viene da un lab» è una decisione da prendere una volta e scrivere, non
da rifare a ogni giro. Un blocco senza attributo significa «non ancora
triato», ed è quello che il rapporto di copertura conta.

## 3. Leggere le tolleranze

Il rapporto stampa quante volte ha usato ciascuna regola:

| Regola | Significato |
|---|---|
| `ESATTO` | il numero coincide |
| `ARROTONDA` | la pagina arrotonda alla precisione che ha scritto |
| `REL_FP` | scarto sotto `1e-4`, confrontato sull'ordine di grandezza |

**`REL_FP > 0` va guardato.** Sono i valori dipendenti da backend e versione
(`max|diff|`, epsilon, zero macchina): a corso pulito oggi sono zero, ma quando
compaiono la pagina deve dichiarare *ordine di grandezza e ambiente*, non un
valore secco. Un numero che passa solo per `REL_FP` non è «verificato»: è
«compatibile con l'ambiente dichiarato», ed è un'affermazione più debole.

## 4. Quando il rapporto elenca un numero

Il rapporto dice cosa c'è in pagina e qual è il valore più vicino prodotto dal
lab. Tre esiti possibili, in ordine di frequenza:

1. **I due numeri si somigliano** → la pagina è rimasta indietro: ricopia
   l'output del lab.
2. **Il numero viene da un calcolo svolto in prosa** → va in `IGNORA`, con la
   motivazione scritta accanto. Ogni voce in `IGNORA` è una rinuncia: si
   aggiunge dopo averla letta, non per far tacere il rapporto.
3. **I due numeri sono lontani e nessuno spiega da dove venga quello in
   pagina** → è un difetto.

## 4b. Il parity fra conti.js e il lab

```
python3 corso/tools/parity_conti.py
```

Da M06 in poi le pagine mostrano, accanto al caso svolto, il ricalcolo per il
profilo del lettore — prodotto da `assets/conti.js` nel browser. Quei numeri
`verifica.py` **non li vede**: non sono nel file HTML.

La garanzia è questa: `conti.js` e `lab_06_1_memoria.py` devono produrre gli
**stessi byte** su otto configurazioni. Uguaglianza esatta, non entro tolleranza —
sono moltiplicazioni fra interi, non c'è arrotondamento da concedere. Esce 1 se
una qualunque voce diverge.

Se fallisce, la pagina sta mostrando al lettore un numero che nessun lab produce.
È il controllo di parità di M02 applicato al corso stesso.

## 5. Cache

`.verifica-cache/` conserva l'output dei lab e si invalida da sola quando il
`.py` cambia. Per forzare la riesecuzione completa:

```
python3 corso/verifica.py --no-cache
```

Il giro completo richiede alcuni minuti: `lab_12_1_eval.py` da solo ne prende
circa cinque.

## 6. Quello che il gate NON copre

- **I rami GPU.** `lab_07_1` sez. 4–5 e `lab_18_1` sez. 5 si fermano puliti
  senza CUDA: il gate non vede i numeri che produrrebbero su hardware vero.
- **I rami legati alla versione.** `enable_gqa` e `F.rms_norm` non esistono su
  `torch 2.2.2`; i percorsi per versioni più recenti non sono mai stati eseguiti.

Per questi due punti c'è **`AMBIENTI.md`**: la matrice di dove il corso è stato
davvero eseguito, e il meccanismo `--registra` / `--confronta` con cui chi ha
l'hardware può riempire le celle vuote. Vanno letti insieme — il gate controlla
che i numeri pubblicati vengano da una misura, la matrice dice *in quale
ambiente* quella misura è stata fatta.
- **La correttezza di quello che il numero significa.** Il gate verifica che il
  numero venga da una misura, non che la misura risponda alla domanda posta.
  Quella parte la fa una revisione, non uno script.
