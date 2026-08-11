// ============================================================
//  PROFILO — l'unico file da modificare per adattare il corso
//            al proprio hardware e al proprio modello.
//
//  Il corso usa numeri VERI, non esempi generici: è quello che
//  lo rende utile.
//
//  COSA FA questo file:
//    - aggiorna l'intestazione di tutte le pagine
//    - fornisce i valori da passare ai lab che accettano parametri,
//      per esempio:  lab_06_1_memoria.py --params 8e9 --gpus 4 --hbm 80
//
//  COSA NON FA, ed è bene saperlo:
//    - NON ricalcola i numeri già scritti nelle pagine HTML. I 54 GB,
//      i 432 GB, le forme e i conteggi LoRA restano quelli del profilo
//      di riferimento. Vanno letti come un caso svolto: il metodo è
//      tuo, i numeri sono suoi.
// ============================================================

const PROFILO = {

  // --- QUESTO PROFILO È TUO O È IL MIO? ---------------------
  // Finché resta false, le pagine dichiarano in chiaro che stai
  // guardando il profilo di RIFERIMENTO — quello su cui il corso è
  // stato scritto — e non il tuo. Mettilo a true quando hai compilato
  // i campi qui sotto col tuo hardware.
  //
  // Non è un dettaglio cosmetico: un lettore che scambia i numeri di
  // un altro per i propri fa esattamente l'errore che il corso esiste
  // per prevenire.
  personalizzato: false,

  // --- Il REGIME, che conta più dei numeri ------------------
  // I quattro regimi non differiscono per quanto sono grandi: cambiano
  // QUALE vincolo morde per primo. -> M18b
  //
  //   "gpu-singola"   una scheda, niente parallelismo. Morde la VRAM.
  //   "multi-gpu"     più schede in un nodo. Morde la VRAM per scheda,
  //                   poi l'interconnessione.
  //   "unificata"     memoria condivisa CPU/GPU (DGX Spark, Grace).
  //                   NON morde la capienza: morde la banda.
  //   "multi-nodo"    cluster con scheduler. Morde la rete fra nodi,
  //                   e la coda.
  //   "noleggio"      GPU effimere (RunPod, Lambda, spot). Morde il
  //                   trasferimento dati e la preemption.
  regime: "multi-gpu",

  // --- Hardware su cui girano i lab -------------------------
  gpu: {
    nome: "H200 NVL",
    quante: 2,
    memoriaGB: 141,          // per singola GPU, NON la somma.
                             // Nel regime "unificata" è invece la
                             // memoria TOTALE condivisa con la CPU.
    interconnessione: "NVLink",
  },
  host: {
    cpu: "EPYC 9555P 64 core",
    ramGB: 1024,
    storage: "NVMe locale",
    so: "RHEL",
  },

  // --- Modello di riferimento usato negli esempi ------------
  // Serve per i conti di memoria (M06), le forme (M04, M09)
  // e il conteggio dei parametri LoRA (M14).
  modello: {
    nome: "denso da 27B",
    parametri: 27e9,
    hidden: 5120,
    intermedio: 13824,
    strati: 46,
    vocabolario: 128000,
    testeQuery: 32,
    testeKV: 8,
  },

  // --- Caso di studio che attraversa tutto il corso ---------
  // Il filo conduttore: un fine-tuning che ha reso meno del previsto,
  // e nessun modo di sapere perché. Cambialo col tuo caso reale.
  caso: {
    titolo: "un LoRA che rende meno del previsto",
    descrizione: "un adattatore LoRA addestrato su un modello denso da 27B, " +
                 "con risultati sotto le aspettative e nessuna diagnosi: " +
                 "dataset, iperparametri, rank, template, maschera della loss, " +
                 "valutazione o disallineamento fra addestramento e inferenza.",

    // La configurazione dell'adattatore del caso di studio.
    // Serve perché M06 e M17 contano la memoria a partire dalla frazione
    // di parametri addestrabili, e quella frazione non è un'assunzione:
    // esce da questi campi. Il conto lo fa lab_14_1_lora.py, sezione 1.
    //   tutti e sette i moduli, r=32  ->  132.644.864 par = 0,491% di 27e9
    // Cambiando r o target_modules cambia la frazione: rifai il conto col
    // lab invece di riusare 0,491%.
    lora: {
      r: 32,
      alpha: 64,                       // convenzione alpha = 2r, vedi M14
      target_modules: ["q_proj", "k_proj", "v_proj", "o_proj",
                       "gate_proj", "up_proj", "down_proj"],
      addestrabili: 132644864,
      frazioneAddestrabili: 0.00491,   // -> lab_06_1_memoria.py --lora-frac
    },
  },
};

// ---- utilità di rendering, usate dalle pagine ---------------

PROFILO.REGIMI = {
  "gpu-singola": "GPU singola",
  "multi-gpu":   "più GPU in un nodo",
  "unificata":   "memoria unificata",
  "multi-nodo":  "cluster multi-nodo",
  "noleggio":    "GPU a noleggio",
};

PROFILO.hwBreve = () => PROFILO.regime === "unificata"
  ? `${PROFILO.gpu.nome} · ${PROFILO.gpu.memoriaGB} GB unificati`
  : `${PROFILO.gpu.quante}× ${PROFILO.gpu.nome} · ${PROFILO.gpu.memoriaGB} GB ciascuna`;

// Nel regime a memoria unificata non esiste una "somma delle memorie":
// c'è un solo pool condiviso con la CPU. Moltiplicarlo per il numero di
// acceleratori darebbe un numero che non esiste.
PROFILO.memoriaTotaleGB = () => PROFILO.regime === "unificata"
  ? PROFILO.gpu.memoriaGB
  : PROFILO.gpu.quante * PROFILO.gpu.memoriaGB;

PROFILO.regimeNome = () => PROFILO.REGIMI[PROFILO.regime] || PROFILO.regime;

document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll("[data-profilo]").forEach(el => {
    const k = el.dataset.profilo;
    if (k === "hw") el.textContent = PROFILO.hwBreve();
    if (k === "modello") el.textContent = PROFILO.modello.nome;
    if (k === "gpu-mem") el.textContent = PROFILO.gpu.memoriaGB;
    if (k === "caso") el.textContent = PROFILO.caso.titolo;
    if (k === "regime") el.textContent = PROFILO.regimeNome();
  });

  // Finché il profilo non è stato personalizzato, ogni pagina lo dichiara.
  // Senza questa riga la sidebar afferma "2× H200 NVL" a un lettore che
  // non ne ha nessuna, e lo lascia credere che i conti siano i suoi.
  if (!PROFILO.personalizzato) {
    document.querySelectorAll(".brand-sub").forEach(el => {
      const nota = document.createElement("span");
      nota.className = "profilo-rif";
      nota.textContent = "profilo di riferimento — non il tuo";
      nota.title = "Compila assets/profilo.js e metti personalizzato: true. Vedi M00b.";
      el.appendChild(nota);
    });
  }
});
