# Auto-revisione — proposte di miglioramento

Scritte **prima** delle revisioni esterne. Nessuna di queste è un finding: il corso non è
sbagliato su questi punti, potrebbe essere migliore.

Budget: oggi 144,5 h, tetto praticabile ~200 h. Le proposte qui sotto sommano **+34 h**,
portando il totale a **178,5 h** se accolte tutte.

---

## M00b — Onboarding: mettere in piedi l'ambiente

**Tipo:** nuovo modulo
**Moduli toccati:** M00, tutti i lab
**Problema che risolve:** il corso è appena stato generalizzato per essere distribuito, ma
non spiega a un nuovo lettore come partire. Non c'è nulla su prerequisiti, installazione,
come adattare `profilo.js`, cosa fare se un lab non gira o se la versione di libreria
differisce.

**Proposta:** un modulo breve prima di M01 che copra:
- prerequisiti reali, dichiarati: Python leggibile ed eseguibile, terminale, niente altro
- verifica dell'ambiente con uno script che stampa versioni e dice cosa manca
- come compilare `assets/profilo.js` con il proprio hardware e modello
- cosa aspettarsi dai lab GPU se non si ha una GPU
- come leggere un errore di versione di libreria invece di arrendersi

**Costo:** +1 h · nuovo totale 145,5 h
**Perché non è un finding:** il corso funziona per chi l'ambiente ce l'ha già. È un problema
di distribuzione, non di correttezza.

---

## Riordino — spostare M19 (MoE) in appendice

**Tipo:** riordino
**Moduli toccati:** M18, M19, M20, M21
**Problema che risolve:** M19 è dichiarato opzionale e non serve al caso di studio, che
riguarda un modello denso. Sta però in mezzo alla sequenza finale, fra parallelismo e
parity, e spezza il filo che porta da «come funziona a scala» a «come si diagnostica».

Un lettore che lo salta, come il corso stesso gli suggerisce, salta un buco in mezzo.

**Proposta:** spostarlo dopo M21 come **appendice A**, rinumerando M20→M19 e M21→M20. La
Parte VI diventa: parallelismo → parity → diagnosi, senza interruzioni. Le appendici
raccolgono le specializzazioni opzionali.

**Costo:** 0 h di lavoro del lettore · totale invariato
**Perché non è un finding:** l'ordine attuale non è sbagliato — MoE dopo parallelismo ha una
sua logica, perché entrambi riguardano la scala. È una scelta di flusso, non di dipendenze.

---

## M09 — spezzare il modulo più lungo in due

**Tipo:** riordino
**Moduli toccati:** M09
**Problema che risolve:** M09 vale 12 h ed è il modulo più lungo del corso. Copre attention,
maschera causale, scala `√d`, GQA, RoPE, RMSNorm, SwiGLU e il blocco completo. Sono otto
argomenti in un modulo solo, e la verifica finale ne controlla cinque.

Un lettore che si ferma a metà non ha un punto di ripresa naturale.

**Proposta:** dividere in **M09a — Attention** (attention, maschera causale, `√d`, GQA: 6 h)
e **M09b — Il blocco completo** (RoPE, normalizzazione, MLP, blocco verificato contro il
riferimento: 6 h). Ciascuno con la propria verifica.

**Costo:** 0 h aggiuntive · totale invariato, cambia solo la granularità
**Perché non è un finding:** il contenuto è corretto e l'ordine interno è giusto. È un
problema di dimensione del blocco, non di sostanza.

---

## Modulo nuovo — Generazione e decoding

**Tipo:** nuovo modulo
**Moduli toccati:** nuovo, fra M12 e M13; referenziato da M17 e M20
**Problema che risolve:** il corso non spiega mai **come un modello produce testo**.
Campionamento, temperatura, top-k, top-p, penalità, criteri di arresto: non compaiono in
nessun modulo.

Il buco si vede in tre punti: M17 parla del costo della generazione senza averla mai
mostrata; M20 usa la temperatura come esempio di disallineamento dandola per nota; e chi
valuta un modello (M12) deve decidere una configurazione di generazione senza sapere cosa
significhi.

È il buco **non dichiarato** più evidente: le esclusioni elencate nell'indice non lo
menzionano, perché me ne sono accorto solo scrivendo M20.

**Proposta:** modulo fra M12 e M13 che copra: il ciclo di generazione un token alla volta,
KV cache (qui esiste davvero, a differenza del training), temperatura, top-k, top-p,
criteri di arresto, e il legame diretto fra configurazione di generazione e riproducibilità
della valutazione. Con lab: stesso modello, stessi logit, cinque configurazioni di
campionamento, distribuzione delle uscite misurata.

**Costo:** +6 h · nuovo totale 150,5 h
**Perché non è un finding:** il corso non afferma niente di falso sulla generazione. Semplicemente
la assume nota dove serve.

---

## M21 — aggiungere un caso di diagnosi svolto per intero

**Tipo:** lab aggiuntivo
**Moduli toccati:** M21
**Problema che risolve:** M21 dà protocollo, scheda e ordine delle ipotesi, ma non mostra
mai una diagnosi completa dall'inizio alla fine. Insegna un metodo senza applicarlo.

*(Questo compare anche nei findings come SHOULD-FIX. Lì il difetto è «il modulo non fa
quello che il corso fa sempre altrove»; qui è la proposta concreta di come colmarlo.)*

**Proposta:** un lab che costruisce un fine-tuning giocattolo **deliberatamente rotto in un
punto noto** — per esempio maschera sfasata di una posizione — e poi esegue il protocollo
di M21 per intero: sintomo misurato con intervallo di confidenza, artefatti inventariati,
ipotesi escluse una alla volta con il criterio scritto prima, verdetto, correzione,
verifica che il sintomo sparisca.

Il lettore vede il metodo funzionare su un caso in cui la risposta è nota, prima di
applicarlo a uno in cui non lo è.

**Costo:** +3 h · nuovo totale 153,5 h
**Perché non è un finding:** vedi nota sopra — la parte «errore» è nei findings.

---

## Percorso ridotto per chi non ha tempo

**Tipo:** riordino / metadato
**Moduli toccati:** index, tutti
**Problema che risolve:** 144,5 h a 4 h a settimana sono nove mesi. È dichiarato onestamente
nell'indice, ma per molti significa non cominciare affatto. E un collaboratore che deve
solo capire *perché il suo fine-tuning non funziona* non ha bisogno di scrivere un
transformer da zero.

**Proposta:** marcare ogni modulo come **nucleo** o **approfondimento**, e proporre nella
pagina iniziale due percorsi:
- **completo** — tutto, 144,5 h
- **diagnostico** — M00, M01, M02, M04, M06, M10, M11, M12, M14, M15, M20, M21: circa 75 h,
  sufficienti a diagnosticare un fine-tuning ma non a scrivere un modello

Il percorso ridotto salta M03, M05, M07, M08, M09, M13, M16–M19: tutto ciò che serve a
*costruire* invece che a *diagnosticare*.

**Costo:** 0 h aggiuntive · offre un'uscita a 75 h
**Perché non è un finding:** il percorso completo non è sbagliato. È che un solo percorso
serve un solo tipo di lettore, e il corso è appena stato aperto a più persone.

---

## Verifica automatica di coerenza fra pagine e lab

**Tipo:** lab aggiuntivo (infrastruttura)
**Moduli toccati:** tutti
**Problema che risolve:** i numeri nelle pagine sono copiati a mano dagli output dei lab, e
durante la scrittura questo ha prodotto almeno tre disallineamenti reali.

*(Anche questo compare nei findings. Lì è il difetto; qui è la forma concreta della
soluzione.)*

**Proposta:** uno script `verifica.py` che esegue tutti i lab, cattura l'output, estrae i
blocchi `<pre><code>` dalle pagine e segnala quelli che non compaiono in nessun output.
Diventa parte del corso — è esso stesso una lezione sul principio che insegna.

**Costo:** 0 h per il lettore (è manutenzione), +0,5 h se lo si presenta come lab in M15,
dove il tema è la riproducibilità · nuovo totale 154 h
**Perché non è un finding:** vedi nota sopra.

---

## Registro degli errori dell'autore come materiale didattico

**Tipo:** nuovo modulo (appendice)
**Moduli toccati:** appendice
**Problema che risolve:** durante la scrittura sono stati commessi e corretti circa
venticinque errori, quasi tutti dello stesso tipo: una regola vera in un regime presentata
come universale. Sono documentati nella cronologia ma non nel corso.

Sono il materiale didattico più onesto disponibile, e riguardano esattamente il modo di
ragionare che il corso vuole insegnare.

**Proposta:** un'appendice che raccolga sei o sette di quegli errori nella forma: cosa avevo
scritto · perché sembrava giusto · cosa lo ha smentito · cosa avrei dovuto fare per
accorgermene prima. Senza morale, senza autocritica: solo il meccanismo.

Chiude il cerchio con M21, dove si insegna che le cause si distinguono con interventi e non
con sintomi — mostrando che chi ha scritto il corso ha sbagliato proprio quello, ripetutamente.

**Costo:** +2 h · nuovo totale 156 h
**Perché non è un finding:** l'assenza di questa appendice non rende il corso scorretto.

---

## Un modulo sui costi economici

**Tipo:** nuovo modulo
**Moduli toccati:** nuovo, dopo M18
**Problema che risolve:** il corso insegna a calcolare la memoria e a misurare il tempo per
passo, ma non collega mai le due cose a **quanto costa** un addestramento — in ore-GPU, in
denaro, in tempo di calendario.

È la domanda che un ingegnere di piattaforma si sente fare per prima da chi decide i
budget, e il corso non gli dà gli strumenti per rispondere.

**Proposta:** modulo breve che parta dai numeri già misurati — secondi per passo (M07),
memoria per rank (M18), dimensione del dataset (M11), numero di epoche — e ricavi ore-GPU,
costo e durata. Con la propagazione dell'incertezza: una stima di costo con un intervallo,
non un numero secco. E il confronto fra le alternative viste nel corso: full FT, LoRA,
QLoRA, DPO, RL, distillazione, ciascuna col proprio costo per punto di miglioramento.

**Costo:** +4 h · nuovo totale 160 h
**Perché non è un finding:** l'esclusione non è dichiarata nell'indice, ma il corso non
afferma nulla di falso sui costi. Semplicemente non ne parla.

---

## Riepilogo

| Proposta | Tipo | Costo |
|---|---|---|
| M00b — Onboarding | nuovo modulo | +1 h |
| M19 in appendice | riordino | 0 |
| M09 spezzato in due | riordino | 0 |
| **Generazione e decoding** | **nuovo modulo** | **+6 h** |
| M21 — caso svolto | lab | +3 h |
| Percorso ridotto a 75 h | metadato | 0 |
| Verifica automatica | infrastruttura | +0,5 h |
| Registro degli errori | appendice | +2 h |
| Costi economici | nuovo modulo | +4 h |
| | **totale** | **+16,5 h → 161 h** |

*(La cifra di +34 h in apertura era una stima a occhio fatta prima di dettagliare le singole
proposte. Il conto vero, sommando la colonna, è +16,5 h. Lascio entrambe visibili perché è
esattamente il tipo di discrepanza che il corso insegna a non lasciar passare.)*

**Se dovessi sceglierne una sola:** *Generazione e decoding*. È l'unico buco non dichiarato
che ho trovato, e si vede da tre moduli diversi che lo assumono noto.
