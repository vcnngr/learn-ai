# Capire, addestrare, diagnosticare

Un corso di **24 moduli / 157 ore** che porta un ingegnere di sistema da zero
assoluto — nessun prerequisito matematico — alla capacità di diagnosticare con
metodo un fine-tuning che rende meno del previsto.

Non insegna a *usare* i modelli. Insegna cosa c'è dentro, e come si capisce
perché un addestramento non ha funzionato.

```
25 pagine HTML autonome · 27 lab Python · nessun framework
```

---

## Il vincolo che tiene in piedi tutto

> **Nessuna affermazione empirica senza la misura accanto.**
> Ogni numero che compare in una pagina deve venire dall'output del lab che lo
> produce.

Non è una dichiarazione d'intenti: è controllata da uno script, e il controllo
fallisce il rilascio.

```bash
python3 corso/verifica.py
```

Esegue i 27 lab, estrae ogni numero dalle pagine e verifica che risalga a un
output reale. Ogni blocco di output è legato al suo lab con `data-lab="..."`, e
il confronto è **per singolo lab**, non sull'unione — una rete larga lascia
passare troppo.

Le tolleranze sono dichiarate, non implicite: `ESATTO`, `ARROTONDA` (alla
precisione scritta in pagina), `REL_FP` (solo per gli scarti sotto `1e-4`, che
dipendono dal backend). Il rapporto stampa quante volte ha usato ciascuna.

Procedura completa in [`corso/RILASCIO.md`](corso/RILASCIO.md).

---

## Revisione multi-modello

Il materiale è stato sottoposto a revisione da modelli diversi, ciascuno con
una lente propria, e i risultati sono versionati in
[`corso/review/`](corso/review/):

| Lente | Domanda guida | Esito |
|---|---|---|
| **Correttezza tecnica** | ogni formula è giusta? ogni numero corrisponde al suo lab? | 25 finding, tutti chiusi |
| **Caccia avversariale** | dove una regola valida in un regime è data per universale? | 12 finding, tutti chiusi |
| **Auto-revisione** | scritta *prima* delle esterne, per confronto | 9 proposte |
| **Solidità pedagogica** | un principiante può seguire questa sequenza? | da fare |

I **findings** e le **proposte** di ciascun revisore sono pubblicati integralmente,
insieme al [confronto fra le tre serie di proposte](corso/review/CONFRONTO_PROPOSTE.md).
Anche quando smentiscono il corso — soprattutto quando lo smentiscono.

Il difetto ricorrente trovato dalle revisioni è sempre lo stesso: **una regola
valida in un regime, presentata come universale.** È diventato il criterio di
lettura del corso intero.

---

## Il tuo hardware, non il mio

Il corso è scritto su un profilo di riferimento — un denso da 27B su due schede —
ma non è legato a quello.

`corso/assets/profilo.js` dichiara hardware, modello e caso di studio. Il campo
che conta di più è `regime`:

| Regime | Cosa morde |
|---|---|
| GPU singola / workstation | la capienza, e basta |
| Più GPU in un nodo | la capienza per scheda, poi la comunicazione |
| Memoria unificata (DGX Spark, Grace) | **la banda**, non la capienza |
| Cluster multi-nodo (HPC) | la rete fra nodi, e la coda |
| GPU a noleggio | il trasferimento dati e la preemption |

Cambiando regime **non cambiano i byte: cambia quale vincolo morde.** È il tema
di M18b, e da M06 in poi ogni tabella di memoria mostra accanto il ricalcolo per
il tuo profilo.

Che le formule JS del ricalcolo coincidano con quelle del lab Python non è un
atto di fede — lo verifica `corso/tools/parity_conti.py` con uguaglianza esatta,
byte per byte, su otto configurazioni.

---

## Da dove si comincia

```bash
git clone <questo-repo>
python3 corso/labs/lab_00_1_ambiente.py   # cosa potrai eseguire TU
open corso/index.html
```

Il primo lab non installa niente: misura l'ambiente e dice quali sezioni del
corso non gireranno su quella macchina, e perché.

**Due percorsi**, entrambi calcolati dall'indice dei moduli e non scritti a mano:

- **completo** — 157 h, tutti i moduli
- **diagnostico** — 89,5 h, i 13 moduli marcati `NUCLEO`: basta a diagnosticare
  un fine-tuning, non a costruire un modello

Serve Python 3.9+, `torch`, `numpy`, `safetensors`. Nessuna GPU fino a M07.

---

## Cosa non è stato verificato

Dichiarato, non nascosto — la matrice completa è in
[`corso/AMBIENTI.md`](corso/AMBIENTI.md).

Il corso è stato scritto su un portatile senza CUDA. Due sezioni non sono **mai
state eseguite**: `lab_07_1` sez. 4–5 e `lab_18_1` sez. 5. Si fermano pulite
senza GPU — *quello* è verificato — ma i numeri che produrrebbero su hardware
vero no. Stessa cosa per i rami che richiedono versioni di PyTorch più recenti
di quella installabile sulla macchina di scrittura.

**Queste non sono lacune coperte da cautele: sono previsioni falsificabili.**
Ogni affermazione non verificata è scritta con accanto come si testa e quale
risultato la smentisce. Una teoria che non dice come si potrebbe smentirla è
un'opinione con dei numeri accanto.

I lab accettano `--registra` e `--confronta`: chi ha l'hardware può registrare
i valori misurati e trasformare «non verificato» in «verificabile da chiunque».

> **Una previsione smentita vale più di dieci confermate:** quella corregge il
> corso, le altre lo lasciano dov'è.

---

## Struttura

```
corso/
  index.html              la mappa e i due percorsi
  m00 … m21               le pagine, autonome, una per modulo
  labs/                   27 script Python, nessun import fra loro
  assets/
    profilo.js            hardware, modello, caso di studio, regime
    conti.js              l'aritmetica della memoria, per il tuo profilo
    nav.js  style.css
  tools/parity_conti.py   conti.js ≡ lab_06_1_memoria.py, byte per byte
  verifica.py             il gate: ogni numero risale a un lab
  RILASCIO.md             checklist da eseguire prima di pubblicare
  AMBIENTI.md             dove il corso è stato eseguito, e dove no
  review/                 findings e proposte dei revisori
```
