# Findings — revisore C (caccia avversariale)

Metodo: eseguiti tutti i 23 lab CPU su torch 2.2.2 + numpy 1.26.4 e confrontato
l'output con ogni numero citato nelle 23 pagine HTML. Le sezioni GPU di M07
(sez. 4–5) e M18 (sez. 5) non sono state eseguite: nessuna CUDA disponibile,
come dichiarato nel brief.

---

### M03 — tabella «La training loss mente»: numeri non corrispondenti al lab
**File:** m03-addestrare.html · **Sezione:** «La training loss mente»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** La tabella riporta valori che il lab non produce. Esempi:
passo 4000: HTML 0.002789 / 0.022094, lab 0.002754 / 0.022311; passo 20000:
HTML 0.001747 / 0.057521, lab 0.001765 / 0.056741; minimo: HTML 0.022094,
lab 0.022311; rapporto finale: HTML «2,6× PEGGIO», lab «2.5x PEGGIO».
Il vincolo del corso è che ogni numero venga dall'output del lab: qui non è così.
Il messaggio qualitativo (la held-out risale mentre la train scende) resta vero.
**Correzione concreta:** rigenerare la tabella copiando l'output corrente di
`labs/lab_03_2_addestramento.py` (seme 0), oppure rieseguire il lab con i
parametri che producevano quei valori e aggiornare il lab. I numeri devono
coincidere cifra per cifra.

### M08 — warmup: il valore a lr=0.5 con warmup è sbagliato di 5×
**File:** m08-optimizer.html · **Sezione:** «Il warmup, misurato»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** La tabella riporta per lr=0.5 con warmup(15) un picco di
**3003.94**; il lab produce **16551.92**. Anche lr=0.2 senza warmup differisce
(HTML 112641.71, lab 112643.40 — lieve). Il primo scarto è troppo grande per
essere rumore di versione: è un numero non prodotto dal lab.
**Correzione concreta:** aggiornare la tabella con l'output corrente del lab.
Il punto didattico (il warmup abbatte il picco di ordini di grandezza) resta
vero anche con 16551.92.

### M13 — «4,6 volte più segnale» e tabella dei token non corrispondono al lab
**File:** m13-cpt-sft.html · **Sezione:** «Il segnale per esempio»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo dice «il continued pretraining estrae 4,6 volte
più segnale per esempio» e la tabella riporta «SFT: 321 token totali, 96
supervisionati». Il lab produce **6,9 volte**, con **214 token totali e 64
supervisionati** (la quota del 30% coincide, quindi il ragionamento regge).
**Correzione concreta:** aggiornare tabella e testo ai valori del lab
(214/64/30%, 6,9×), o correggere il lab se i valori attesi erano altri.

### M13 — perplessità dopo SFT: 676 nel testo, 694 nel lab
**File:** m13-cpt-sft.html · **Sezione:** «Primo risultato scomodo»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo riporta «la perplessità passa da 1,08 a 676» e
la tabella «solo SFT: 87506.30». Il lab produce **694.15** e **87560.12**.
Il lab è deterministico (seed fissati), quindi i numeri dovrebbero coincidere
esattamente.
**Correzione concreta:** aggiornare testo e tabella all'output del lab.

### M09 — il valore 2.384e-07 è attribuito anche ai confronti dove non vale
**File:** m09-transformer.html · **Sezioni:** «Attention» e ripetuto nel sottotitolo
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo cita `max|diff| = 2.384e-07` per «attention piena
vs F.scaled_dot_product_attention» e per «attention causale». Il lab produce
**1.192e-07** per entrambe; 2.384e-07 è il valore del **blocco completo**
(sezione 8). Il sottotitolo del modulo («il risultato non è "hai capito" ma
max|diff| = 2.384e-07 → OK») è quindi corretto solo per il blocco.
**Correzione concreta:** distinguere i valori: 1.192e-07 per le due attention,
2.384e-07 per il blocco. Entrambi sono rumore di fp32, quindi il punto regge.

### M14 — merge: max|diff| citato non corrisponde al lab
**File:** m14-lora.html · **Sezione:** «Il merge, dimostrato»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo riporta `max|diff| = 8.882e-16` (nella risposta
iniziale anche «8.9·10⁻¹⁶»). Il lab produce **2.220e-15**. Entrambi sono zero
macchina, ma il vincolo del corso è la corrispondenza esatta con l'output.
**Correzione concreta:** aggiornare a 2.220e-15 (valore del lab, seed 0).

### M15 — i byte del dataset d'esempio non corrispondono al lab
**File:** m15-finetuning.html · **Sezione:** «Perché l'impronta dei dati»
**Severità:** NITPICK
**Cosa è sbagliato:** Il testo riporta «58 byte» e «59 byte»; il lab produce
**60 e 61 byte** (le stringhe nel lab sono più lunghe di quelle citate).
La differenza di un carattere — che è il punto didattico — è corretta.
**Correzione concreta:** aggiornare a 60/61 byte, o accorciare le stringhe
del lab per far tornare 58/59.

### M02 — «23 volte su 40» non corrisponde a nessuna riga dello sweep
**File:** m02-gradienti.html · **Sezione:** Controllo A, nota «Come non si
diagnostica»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo dice «riducendola di venticinque volte il segno
diventa testa o croce — 23 volte su 40 la loss scende». La perturbazione forte
nel lab è scala 0.5; 25 volte meno è 0.02, dove lo sweep produce **15/40**
negativi. A scala 0.005 (100× meno) produce **22/40**. Il numero 23 non compare
in nessuna riga. Il punto qualitativo (il segno non è un criterio) è corretto e
ben supportato; il numero specifico no.
**Correzione concreta:** citare la riga reale: «riducendola di venticinque
volte, 15 volte su 40 la loss scende; a cento volte, 22 su 40».

### M15 — la sezione 2 del lab non è riproducibile, ma il testo cita numeri specifici
**File:** m15-finetuning.html + labs/lab_15_1_riproducibile.py · **Sezione:**
«L'errore più comune»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** Il testo cita «esecuzione #1: 1.8e-01, #2: 2.1e-01,
#3: 1.9e-01». La sezione 2 del lab usa `random.randrange(10**6)` **senza fissare
il seme**, quindi produce valori diversi a ogni esecuzione (verificato: due run
hanno dato 5.8e-01/4.3e-01/1.1e-01 e 6.0e-01/1.0e-01/6.7e-02). I numeri citati
non sono riproducibili per costruzione — proprio nel modulo sulla
riproducibilità. È un difetto di tipo 4: il lab non può produrre i numeri del
testo.
**Correzione concreta:** fissare il seme del generatore `random` nella sezione
2 del lab (es. `rng = random.Random(0)` al posto dell'uso globale) e aggiornare
il testo coi valori risultanti. Il punto didattico resta identico e diventa
riproducibile.

### M06 ↔ M14 — lo 0,5% di addestrabili non corrisponde a nessuna configurazione del corso
**File:** m06-memoria.html · **Sezione:** «Dove va il risparmio» (e M17,
m17-rl.html, tabella dei GB statici)
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** M06 calcola LoRA con «0,5% addestrabili» (135M parametri,
55.9 GB statici). Ma in M14 le configurazioni presentate danno: q,v a r=16 →
0.045%; q,k,v,o → 0.091%; solo MLP → 0.155%; tutti e sette → 0.246%. Per
arrivare a 0.5% servirebbero tutti e sette i moduli a r≈33, o una
configurazione non presentata. Lo stesso 0.5% ritorna in M17 (tabella SFT/DPO/
PPO/GRPO). Il lettore che prova a riprodurre il conto di M06 con gli strumenti
di M14 non ci riesce: è una contraddizione interna fra moduli.
**Correzione concreta:** o dichiarare in M06/M17 quale configurazione produce
0.5% (es. «tutti e sette i moduli a r=32»), o rifare i conti con una delle
configurazioni di M14. Il primo approccio costa una riga.

### M16 — la spiegazione del perché logp(preferita) scende è corretta ma incompleta
**File:** m16-dpo.html · **Sezione:** «Il risultato che sorprende»
**Severità:** SHOULD-FIX
**Cosa è sbagliato:** (punto debole #5 del brief). Il testo spiega che la loss
impone solo la differenza, quindi «nulla spinge verso l'alto in senso
assoluto». Vero, ma non risponde alla domanda che il lettore si fa: *perché*
l'ottimizzazione sceglie proprio quella direzione? Mancano due fatti
disponibili nel lab stesso: (1) la preferita parte già vicino al suo massimo
possibile (logp −0.028, quasi saturata) mentre la rifiutata parte da −56 e ha
enorme spazio per scendere — l'asimmetria dello spazio disponibile rende la
direzione «giù entrambe» quella a minor resistenza; (2) il gradiente della
loss rispetto a logp(preferita) ha segno che la spingerebbe *in su*, ma con
peso σ(−β·margine) che diventa piccolissimo non appena il margine cresce. La
spiegazione attuale è corretta come analisi della formula, ma non della
dinamica osservata.
**Correzione concreta:** aggiungere due righe: la preferita è quasi satura
all'inizio (poco da guadagnare salendo), la rifiutata ha decine di nat da
perdere; e il peso del gradiente che alzerebbe la preferita si annulla
appena il margine cresce. Entrambi i fatti sono leggibili nei numeri già
stampati dal lab.

### index.html — «I moduli in grigio non sono ancora scritti», ma non ce ne sono
**File:** index.html · **Sezione:** «La mappa»
**Severità:** NITPICK
**Cosa è sbagliato:** Il testo annuncia moduli in grigio non ancora scritti, ma
in `assets/nav.js` tutti i 22 moduli hanno `ok: 1`. La frase è un residuo.
**Correzione concreta:** togliere la frase o renderla condizionale alla presenza
di moduli con `ok: 0`.

---

## Note sui punti deboli dichiarati dall'autore (verificati)

1. **M02 log(V)**: i limiti di validità sono dichiarati correttamente in tre
   punti della pagina e nel lab. Non ho trovato casi non coperti. OK.
2. **M06/M08 AdamW 8 byte/parametro, 4× i pesi in bf16**: corretto, misurato
   dal lab 08.1 su optimizer veri. OK.
3. **M06/M07 k=10**: dichiarato esplicitamente come stima non universale in
   entrambi i moduli, con procedura di misura in M07. OK.
4. **M09 RoPE**: la proprietà «dipende solo da m−n» è dimostrata dal lab e il
   caveat sull'estrapolazione è presente («Cosa NON segue da qui»). OK.
5. **M16 DPO**: vedi finding dedicato sopra — incompleta, non sbagliata.
6. **M19 MoE**: l'onestà sull'esperimento imposto è esplicita sia nel testo
   sia nel lab. I limiti sono dichiarati. OK.
7. **M14 QLoRA 0,531 byte/parametro, ~840 MB di scale su 27B**: verificato —
   27e9 × 2 byte / 64 = 843,75 MB ≈ 840 MB. OK.
8. **M20 «3000 volte»**: 7.727e-03 / 2.384e-06 = 3241 ≈ 3000. Il framing è
   corretto perché il testo dichiara subito «su un singolo strato è piccolo;
   su decine si accumula». OK.
9. **M12 regola del tre, 1,5%**: 3/200 = 0.015. Corretto. OK.
10. **M21 ordinamento per costo**: giudizio dichiarato come tale («È un
    giudizio dell'autore, non una misura» nel brief; nel modulo l'ordine è
    motivato). OK.

## Cosa NON ho potuto verificare

- Lab GPU di M07 sez. 4–5 e M18 sez. 5: nessuna CUDA sulla macchina di review.
  Il codice si ferma con messaggio esplicito come promesso (verificato), ma la
  sua correttezza numerica su GPU resta **non verificabile** qui.
- Rami versione-specifici: su torch 2.2.2 il ramo `enable_gqa=True` non esiste
  e il lab lo dichiara correttamente; `F.rms_norm` assente e il fallback è
  usato (max|diff| = 0.000e+00 contro la formula di riferimento). I rami per
  versioni più recenti sono **mai eseguiti** — non verificabile.
- Le affermazioni dipendenti dalla versione di libreria (chat template nella
  config del tokenizer, default TF32, dettagli DoRA/ORPO) sono tutte marcate
  nel testo come da verificare sul proprio stack. Conforme al brief.
