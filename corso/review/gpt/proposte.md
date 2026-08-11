# Proposte — revisore A

## Pronuncia sulle tre domande dell'indice

- **Ordine:** difendibile. M12 prima di M13–M17 crea il prerequisito di misura; M20 prima di M21 fornisce il primo controllo economico della diagnosi. Non propongo riordino.
- **Durate:** **non verificabile** dal corpus. Mancano tempi osservati su lettori del profilo dichiarato; non posso sostenere che M09 12 h, M13 4 h o altri moduli siano gravemente sottostimati.
- **Buchi:** nessun buco di contenuto non dichiarato dimostrabile con questa revisione. Esiste invece un buco di verifica: rami CUDA e rami per versioni PyTorch più recenti non sono mai stati eseguiti. Non richiede necessariamente un nuovo modulo.

## Controllo automatico lab → HTML

**Tipo:** altro  
**Moduli toccati:** M01–M21  
**Problema che risolve:** impedisce che i numeri pubblicati divergano dagli output dei lab dopo una modifica al codice, ai dati o all'ambiente.  
**Proposta:** aggiungere al processo di build un controllo che esegua i 24 lab, catturi i valori destinati alle pagine e confronti i blocchi HTML. Usare uguaglianza per interi/valori deterministici e tolleranze dichiarate più ambiente per risultati floating-point dipendenti dal backend. Le sezioni intenzionalmente casuali devono usare una lista di seed espliciti oppure non pubblicare valori fissi. Il controllo deve fallire indicando pagina, campo, atteso e osservato.  
**Costo:** 0 h aggiuntive per il lettore; totale invariato a 144,5 h.  
**Perché non è un finding:** l'assenza di automazione non rende falso il corso; le divergenze attuali sono già registrate nei findings.

## Separare gradiente diretto e aggiornamento parametrico in DPO

**Tipo:** lab aggiuntivo  
**Moduli toccati:** M16  
**Problema che risolve:** rende verificabile perché la derivata rispetto a `logp(preferita)` punta verso l'alto mentre, nel modello con parametri condivisi, la sua misura assoluta può scendere.  
**Proposta:** aggiungere un esercizio in due parti: prima calcolare con autograd la loss DPO su due log-probabilità scalari indipendenti e stampare i segni delle derivate; poi eseguire un update dell'attuale modello condiviso e rimisurare entrambe le log-probabilità. Chiudere confrontando «direzione locale della loss» e «esito dopo la proiezione nello spazio dei parametri».  
**Costo:** 0,5 h; nuovo totale 145 h.  
**Perché non è un finding:** corretta la spiegazione matematica, il modulo può essere tecnicamente valido anche senza questo esercizio; il lab renderebbe il passaggio più controllabile.

## Matrice di compatibilità per i rami non eseguiti

**Tipo:** altro  
**Moduli toccati:** M07, M09, M18  
**Problema che risolve:** distingue incompatibilità di versione da errori del corso nei percorsi oggi non verificabili.  
**Proposta:** conservare output e stato di esecuzione degli stessi lab su `torch 2.2.2`, su una versione più recente che attivi `enable_gqa` e `F.rms_norm`, e su almeno un ambiente CUDA per M07/M18. Registrare versione PyTorch/CUDA, GPU, dtype e comando. Nessun benchmark prestazionale va trasferito fra ambienti: il risultato richiesto è pass/fail dei rami e coerenza delle formule.  
**Costo:** 0 h aggiuntive per il lettore; nuovo totale invariato a 145 h dopo la proposta precedente.  
**Perché non è un finding:** non supportare più versioni o hardware non è di per sé errato; è un miglioramento della copertura. I guasti già visibili nel codice GPU sono findings separati.
