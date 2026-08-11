// ============================================================
//  conti.js — l'aritmetica della memoria, per il TUO profilo.
//
//  Le pagine mostrano i numeri del profilo di riferimento, ed è
//  giusto così: sono un caso svolto, verificato riga per riga da
//  verifica.py contro l'output dei lab. Questo file serve a mettere
//  accanto a quei numeri i tuoi.
//
//  ATTENZIONE, ed è il punto di tutto il file:
//  queste formule devono essere le STESSE di labs/lab_06_1_memoria.py.
//  Non "equivalenti", non "aggiornate": le stesse. Se divergono, la
//  pagina mostra al lettore un numero che nessun lab produce, ed è
//  esattamente il difetto che il corso esiste per impedire.
//
//  Per questo esiste tools/parity_conti.py: esegue le due strade sullo
//  stesso profilo e confronta voce per voce, con esito binario. Sta
//  nella checklist di rilascio (RILASCIO.md), non è facoltativo.
// ============================================================

const CONTI = {};

const GB = 1e9;

// byte per parametro ADDESTRABILE, per optimizer.
// Stessa tabella di lab_06_1_memoria.py.
CONTI.STATI = { adamw: 8, adamw8bit: 2, sgd: 4 };

/**
 * Voci statiche di memoria, in byte.
 * Porto di calcola() da lab_06_1_memoria.py — stessa struttura,
 * stesse assunzioni, stesso ordine delle voci.
 */
CONTI.voci = function (P, metodo, opt, loraFrac) {
  let wDtype, addestrabili;
  if (metodo === "full") {
    wDtype = 2; addestrabili = P;                 // pesi bf16
  } else if (metodo === "lora") {
    wDtype = 2; addestrabili = P * loraFrac;
  } else {                                         // qlora
    wDtype = 0.5; addestrabili = P * loraFrac;     // base quantizzata a 4 bit
  }

  const stati = CONTI.STATI[opt];

  const voci = [
    { nome: "pesi (base)",      byte: P * wDtype },
    { nome: "gradienti",        byte: addestrabili * 2 },
    { nome: "stati optimizer",  byte: addestrabili * stati },
  ];
  // Con full fine-tuning la copia master segue TUTTI i parametri;
  // con LoRA/QLoRA solo gli addestrabili. È la differenza che rende
  // LoRA conveniente, e va tenuta identica al lab.
  voci.push({
    nome: "copia master fp32",
    byte: (metodo === "full" ? P : addestrabili) * 4,
  });
  return voci;
};

CONTI.totale = (voci) => voci.reduce((s, v) => s + v.byte, 0);

/**
 * Ci sta? La risposta dipende dal REGIME, non solo dai byte.
 *
 * Non è una raffinatezza: nei quattro regimi cambia proprio la domanda.
 * Su GPU discrete il vincolo è la capienza per scheda; su memoria
 * unificata la capienza è una sola e molto più grande, e a mordere è
 * la banda — che questa funzione NON stima, perché non si deduce
 * dall'aritmetica. Dove non so, restituisco "non calcolabile" invece
 * di un numero plausibile.
 */
CONTI.capienza = function (totByte, profilo) {
  const tot = totByte / GB;
  const perScheda = profilo.gpu.memoriaGB;
  const n = profilo.gpu.quante;
  const r = profilo.regime;

  const righe = [];

  if (r === "unificata") {
    righe.push({
      nome: "pool unificato",
      gb: tot,
      tetto: perScheda,
      entra: tot < perScheda,
      nota: "un solo pool condiviso con la CPU: non si somma e non si partiziona",
    });
    return {
      righe,
      avvertenza: "In questo regime la capienza è la domanda facile. Quella " +
        "che decide è la BANDA: la stessa memoria che ti fa entrare il modello " +
        "è molto più lenta della HBM di una scheda discreta. Il tempo per passo " +
        "va misurato, non dedotto — questo conto non lo stima.",
    };
  }

  if (r === "gpu-singola") {
    righe.push({
      nome: "tutto su una scheda",
      gb: tot, tetto: perScheda, entra: tot < perScheda,
      nota: "niente partizionamento disponibile",
    });
    return {
      righe,
      avvertenza: "Con una sola scheda le strategie di partizionamento di M18 " +
        "non si applicano: se non ci sta, le leve sono quantizzare la base " +
        "(QLoRA), ridurre gli addestrabili, o le attivazioni (batch, sequenza, " +
        "checkpointing).",
    };
  }

  // multi-gpu, multi-nodo, noleggio: schede discrete, si può partizionare
  righe.push({
    nome: "replicato per GPU (DDP)",
    gb: tot, tetto: perScheda, entra: tot < perScheda,
    nota: "DDP non riduce la memoria, la replica",
  });
  righe.push({
    nome: "partizionato (ZeRO-3, ideale)",
    gb: tot / n, tetto: perScheda, entra: tot / n < perScheda,
    nota: "limite teorico: divisione perfetta, buffer di comunicazione ignorati",
  });

  let avvertenza = "La somma su tutte le GPU non è un pool unico: ogni scheda " +
    "ha il suo tetto, e il conto va fatto per scheda.";
  if (r === "multi-nodo") {
    avvertenza += " Su più nodi la riga ZeRO-3 è ottimistica di più: " +
      "l'all-gather dei pesi attraversa la rete fra nodi, non l'interconnessione " +
      "interna, e lì il costo cambia di ordine di grandezza.";
  }
  if (r === "noleggio") {
    avvertenza += " Su GPU a noleggio l'aritmetica è la stessa, ma il vincolo " +
      "che ti ferma spesso non è questo: sono il trasferimento del dataset e " +
      "dei checkpoint, e la possibilità che l'istanza venga tolta a metà run.";
  }
  return { righe, avvertenza };
};

/** Conto completo, pronto da stampare. */
CONTI.perProfilo = function (profilo, metodo, opt) {
  const P = profilo.modello.parametri;
  const loraFrac = profilo.caso && profilo.caso.lora
    ? profilo.caso.lora.frazioneAddestrabili : 0.00491;
  const voci = CONTI.voci(P, metodo, opt, loraFrac);
  const tot = CONTI.totale(voci);
  return {
    parametri: P, metodo, optimizer: opt, loraFrac,
    voci, totaleByte: tot, totaleGB: tot / GB,
    capienza: CONTI.capienza(tot, profilo),
  };
};

CONTI.GB = GB;

// ------------------------------------------------------------
// Rendering: riempie ogni <div data-conto="metodo"> con il conto
// per il profilo corrente. Il blocco statico accanto NON viene
// toccato: quello è il caso svolto, ed è verificato dal gate.
// ------------------------------------------------------------

CONTI.render = function (profilo) {
  const box = document.querySelectorAll("[data-conto]");
  if (!box.length || typeof profilo === "undefined") return;

  box.forEach(el => {
    const metodo = el.dataset.conto || "full";
    const opt = el.dataset.optimizer || "adamw";
    const c = CONTI.perProfilo(profilo, metodo, opt);
    const fmt = g => g >= 10 ? g.toFixed(1) : g.toFixed(2);

    let h = `<div class="tuo-head">
        <strong>Il tuo profilo</strong>
        <span>${profilo.modello.nome} · ${profilo.hwBreve()} · ${profilo.regimeNome()}</span>
      </div>`;

    h += `<table class="tuo-tab"><tr><th>voce</th><th>GB</th></tr>`;
    c.voci.forEach(v => {
      h += `<tr><td>${v.nome}</td><td>${fmt(v.byte / GB)}</td></tr>`;
    });
    h += `<tr class="tot"><td>STATICO totale</td><td>${fmt(c.totaleGB)}</td></tr></table>`;

    h += `<table class="tuo-tab"><tr><th>ci sta?</th><th>GB</th><th>tetto</th><th></th></tr>`;
    c.capienza.righe.forEach(r => {
      h += `<tr><td>${r.nome}<br><span class="nota">${r.nota}</span></td>
            <td>${fmt(r.gb)}</td><td>${r.tetto}</td>
            <td class="${r.entra ? "si" : "no"}">${r.entra ? "ENTRA" : "NON ENTRA"}</td></tr>`;
    });
    h += `</table><p class="tuo-avv">${c.capienza.avvertenza}</p>`;

    h += `<p class="tuo-fonte">Calcolato da <code>assets/conti.js</code> sui campi di
      <code>assets/profilo.js</code>, con le stesse formule di
      <code>lab_06_1_memoria.py</code> — l'uguaglianza fra le due strade è
      controllata da <code>tools/parity_conti.py</code>.
      ${profilo.personalizzato
        ? ""
        : `<strong>Ma il profilo non è ancora il tuo:</strong> finché
           <code>personalizzato</code> resta <code>false</code>, questi numeri
           coincidono con quelli del caso svolto qui sopra.`}</p>`;

    el.innerHTML = h;
    el.classList.add("tuo-conto");
  });
};

if (typeof document !== "undefined") {
  document.addEventListener("DOMContentLoaded", () => {
    if (typeof PROFILO !== "undefined") CONTI.render(PROFILO);
  });
}

if (typeof module !== "undefined" && module.exports) module.exports = CONTI;
