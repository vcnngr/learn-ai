#!/usr/bin/env python3
"""
Lab 21.1 — Il protocollo di diagnosi.   CPU.

    python3 corso/labs/lab_21_1_diagnosi.py
    python3 corso/labs/lab_21_1_diagnosi.py --scheda    # stampa la scheda vuota

Non insegna una tecnica nuova: mette in ordine quelle dei venti moduli
precedenti, per rapporto informazione/costo.
"""

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--scheda", action="store_true", help="stampa solo la scheda da compilare")
args = ap.parse_args()


def sez(t):
    print(f"\n{'═' * 74}\n{t}\n{'═' * 74}")


# ======================================================================
IPOTESI = [
    # (ordine, nome, costo, esperimento che la FALSIFICA, criterio di decisione, modulo)
    (1, "Disallineamento addestramento/inferenza",
     "ore, nessun addestramento",
     "Scala di parity: base -> adattatore all'init -> merged -> converted -> quantized.\n"
     "     Confronta ogni passo col PRECEDENTE sullo stesso insieme di prompt.",
     "I primi tre confronti devono dare scarto ~0 (rumore numerico).\n"
     "     Il primo che sfora e' il colpevole: ti fermi li'.",
     "M20"),

    (2, "Adattatore non neutro all'inizializzazione",
     "minuti",
     "Carica l'adattatore appena inizializzato e misura la loss sullo stesso\n"
     "     batch del modello base, in eval(), stesso seme.",
     "Deve dare scarto ESATTAMENTE nullo: con B=0 l'adattatore non\n"
     "     contribuisce. Qualunque scarto = configurazione sbagliata.",
     "M02, M14"),

    (3, "Chat template diverso fra addestramento e base",
     "ore",
     "Confronta con repr() il TESTO renderizzato dal tuo percorso di\n"
     "     addestramento con quello prodotto dal template del modello base,\n"
     "     a parita' di input strutturato.",
     "Devono coincidere carattere per carattere. Se differiscono, ti\n"
     "     fermi qui: le ipotesi successive non sono interpretabili.",
     "M10, M02"),

    (4, "Maschera della loss sbagliata o sfasata",
     "ore",
     "Stampa (token, etichetta) affiancati per 3 esempi reali. Verifica\n"
     "     QUALI posizioni contribuiscono, non quante.",
     "Le posizioni attive devono essere esattamente la risposta, piu' il\n"
     "     terminatore di turno. Se ci trovi il prompt, o manca l'EOS, hai\n"
     "     trovato il problema.",
     "M10, M11"),

    (5, "La valutazione non misura quello che credi",
     "giorni, nessun addestramento",
     "Bootstrap sull'insieme di eval, confronto APPAIATO base vs fine-tuned.\n"
     "     Piu' un controllo di contaminazione fra train ed eval.",
     "Se l'IC 95% della differenza contiene lo zero, non hai dimostrato\n"
     "     nulla - ne' in un senso ne' nell'altro. Serve piu' eval, non\n"
     "     piu' addestramento.",
     "M12, M11"),

    (6, "Batch effettivo o learning rate diversi da quelli creduti",
     "ore",
     "Ricalcola micro x accumulazioni x GPU. Verifica che la loss sia\n"
     "     divisa per il numero di accumulazioni prima del backward.",
     "Se la divisione manca, il learning rate effettivo e' moltiplicato\n"
     "     per le accumulazioni. Confronta col valore che credevi.",
     "M08"),

    (7, "Regime sbagliato: conoscenza contro comportamento",
     "giorni",
     "Il modello sbaglia il CONTENUTO o il FORMATO? Misura le due cose\n"
     "     separatamente su esempi in cui il formato e' banale.",
     "Formato giusto e contenuto sbagliato -> serve continued pretraining,\n"
     "     non SFT. Nessun iperparametro corregge un regime sbagliato.",
     "M13"),

    (8, "Rango insufficiente per il compito",
     "giorni, piu' addestramenti",
     "Ripeti l'addestramento a r crescente tenendo alpha/r COSTANTE.\n"
     "     Valuta ogni run con l'eval di M12, con intervalli.",
     "Se la metrica migliora ancora all'ultimo r provato, il rango e' il\n"
     "     vincolo. Se e' piatta, NON lo e' e aumentarlo e' spreco.",
     "M14, M12"),

    (9, "Dati insufficienti o di bassa qualita'",
     "settimane",
     "Curva di scala: addestra al 25%, 50%, 100% degli esempi e valuta\n"
     "     tutti sullo stesso insieme tenuto da parte, con intervalli.",
     "Ancora in salita al 100% -> servono piu' dati. Piatta fra 50% e\n"
     "     100% -> il problema NON e' la quantita'.",
     "M11, M12"),
]


def stampa_ipotesi():
    sez("Le ipotesi, in ordine di rapporto informazione/costo")
    print("""  L'ordine NON e' per probabilita': e' per COSTO. Un'ipotesi che si
  esclude in ore va provata prima di una che costa settimane, anche se
  sembra meno probabile.

  Regola vincolante: ogni ipotesi si esclude PRIMA di passare alla
  successiva. Un test eseguito mentre un'ipotesi precedente e' ancora
  aperta non isola niente - e' la scala di M02.
""")
    for n, nome, costo, esp, crit, mod in IPOTESI:
        print(f"  {'─' * 70}")
        print(f"  {n}. {nome}")
        print(f"     costo: {costo}   ({mod})")
        print(f"     ESPERIMENTO   {esp}")
        print(f"     DECISIONE     {crit}")


def stampa_scheda():
    sez("Scheda di diagnosi — da compilare PRIMA di eseguire")
    print("""  Il criterio di decisione va scritto prima del test. Se lo scrivi dopo
  aver visto il risultato, non stai falsificando un'ipotesi: la stai
  interpretando.

  ---------------------------------------------------------------------
  SINTOMO OSSERVATO
    che cosa fa il modello di sbagliato, in concreto:  ___________________
    su quale insieme e' stato misurato:                ___________________
    con quale metrica, e quanto vale:                  ___________________
    valore della stessa metrica sul modello BASE:      ___________________
    intervallo di confidenza della differenza:         ___________________

  Se l'ultima riga contiene lo zero, FERMATI: non hai ancora un
  fenomeno da spiegare. Vai a M12.

  ---------------------------------------------------------------------
  ARTEFATTI DISPONIBILI                                    si / no
    manifest dell'addestramento                            ______
    sha256 del dataset                                     ______
    revisione esatta del modello base                      ______
    template renderizzato salvato come testo               ______
    semi (pesi, ordine dati, dropout)                      ______
    checkpoint valutato, identificato senza ambiguita'     ______
    configurazione di inferenza usata in valutazione       ______

  Ogni 'no' RIMUOVE ipotesi dalla lista dei verificabili. Segnale quali.

  ---------------------------------------------------------------------
  PER OGNI IPOTESI, PRIMA DI ESEGUIRE

    ipotesi n.: ______   nome: ____________________________________

    esperimento minimo che la falsifica:
      ___________________________________________________________

    che risultato la ESCLUDE:      ____________________________________
    che risultato la CONFERMA:     ____________________________________
    costo stimato:                 ____________________________________

    [ ] criterio scritto PRIMA di eseguire
    [ ] tutte le ipotesi precedenti gia' escluse

    risultato osservato:           ____________________________________
    verdetto:  esclusa / confermata / non determinabile
    se non determinabile, che artefatto mancava: _____________________
  ---------------------------------------------------------------------""")


def stampa_esiti():
    sez("I tre esiti possibili, e sono tutti accettabili")
    print("""  1. CAUSA IDENTIFICATA
     Un'ipotesi confermata da un esperimento che l'avrebbe potuta
     falsificare. Si corregge, si rifa' il run, si verifica che il
     sintomo sparisca. Se non sparisce, la causa era un'altra: torna
     alla lista.

  2. CAUSE ELIMINATE, NESSUNA CONFERMATA
     Non e' un fallimento. Hai ristretto lo spazio e sai cosa NON e'.
     Il risultato utile e': lista aggiornata, evidenza raccolta, e il
     prossimo esperimento discriminante identificato.

  3. NON DETERMINABILE
     Mancano artefatti, il run non e' riproducibile, o piu' cause sono
     compatibili con l'evidenza. La conclusione onesta e': "si rifa'
     tracciando tutto" - e la prima cosa da produrre e' il manifest
     di M15, non un altro addestramento alla cieca.

  Un esito 2 o 3 dichiarato esplicitamente vale piu' di un esito 1
  inventato. La diagnosi non e' un rito che deve finire con un colpevole.""")


def stampa_antipattern():
    sez("Come NON si fa (e perche' sembra ragionevole)")
    print("""  CAMBIARE PIU' COSE INSIEME
    "Alzo il rango, abbasso il learning rate e aggiungo dati."
    Se migliora non sai quale delle tre. Se peggiora nemmeno.
    Sembra efficiente perche' risparmia run. Costa piu' tempo.

  PARTIRE DALL'IPOTESI PIU' INTERESSANTE
    Il rango e gli iperparametri sono piu' interessanti di un a-capo
    nel template. Ma l'a-capo si esclude in un'ora e il rango in una
    settimana. L'ordine e' per COSTO, non per curiosita'.

  LEGGERE IL RISULTATO E POI DECIDERE COSA SIGNIFICA
    Senza criterio scritto prima, qualunque risultato conferma
    l'ipotesi che avevi in mente. E' il motivo per cui la scheda
    chiede di scriverlo prima.

  FIDARSI DELLA TRAINING LOSS
    Scende in tutti gli scenari rotti che hai costruito nel corso:
    template sbagliato (M10), maschera sfasata (M11), regime
    sbagliato (M13), DPO che peggiora il modello (M16).

  DEDURRE LA CAUSA DAL SINTOMO
    Le cause producono sintomi che si somigliano. Si distinguono con
    un INTERVENTO che ne elimina una alla volta, non con una tabella
    sintomo -> causa. E' la lezione di M02, e vale qui piu' che altrove.""")


def stampa_mappa():
    sez("Dove ogni strumento e' stato costruito")
    print(f"  {'strumento':<52} {'modulo':>8}")
    for s, m in [
        ("leggere i pesi di un checkpoint", "M04"),
        ("calcolare la memoria prima di lanciare", "M06"),
        ("misurare il termine statico e le attivazioni", "M07"),
        ("diagnosticare una loss curve", "M08"),
        ("verificare un'implementazione contro il riferimento", "M09"),
        ("ispezionare template e maschera token per token", "M10"),
        ("validare un dataset, dedup, contaminazione", "M11"),
        ("intervalli di confidenza, confronto appaiato", "M12"),
        ("distinguere conoscenza da comportamento", "M13"),
        ("contare i parametri LoRA, verificare il merge", "M14"),
        ("manifest e riproducibilita'", "M15"),
        ("leggere le metriche di un DPO", "M16"),
        ("scegliere fra RL e distillazione", "M17"),
        ("memoria per rank e scaling", "M18"),
        ("metriche di routing MoE", "M19"),
        ("scala di parity", "M20"),
    ]:
        print(f"  {s:<52} {m:>8}")
    print("""
  Nessuno di questi e' nuovo. Questo modulo non aggiunge strumenti:
  aggiunge l'ORDINE in cui usarli, che e' la cosa che distingue una
  diagnosi da una serie di tentativi.""")


# ======================================================================
if args.scheda:
    stampa_scheda()
else:
    stampa_ipotesi()
    stampa_scheda()
    stampa_esiti()
    stampa_antipattern()
    stampa_mappa()
print()
