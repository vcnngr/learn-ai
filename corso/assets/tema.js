// ============================================================
//  tema.js — selettore del tema, tre posizioni.
//
//  NON un bottone che cicla. Un ciclo nasconde le opzioni, non dice
//  cosa viene dopo e cambia larghezza a ogni clic: va bene per un
//  interruttore binario, non per tre posizioni.
//
//  Qui le tre posizioni si vedono tutte insieme e quella attiva e'
//  marcata da un filetto SOPRA, in `signal`. Non e' un ornamento: e'
//  l'idioma che le .note del corso gia' usano per dichiarare di che
//  natura e' una cosa. Il comando dichiara il proprio stato con la
//  stessa grammatica con cui le pagine dichiarano il proprio.
//
//  "sistema" e non "auto": dice CHI comanda, che e' la disciplina del
//  resto del corso.
//
//  La scelta si applica in <head>, prima del primo paint, con lo
//  snippet inline che ogni pagina porta. Se stesse qui, la pagina
//  comparirebbe chiara e poi diventerebbe scura.
// ============================================================

(function () {
  const STATI = [
    { id: "auto",   voce: "sistema", desc: "segue la preferenza del sistema" },
    { id: "chiaro", voce: "chiaro",  desc: "chiaro, anche se il sistema è scuro" },
    { id: "scuro",  voce: "scuro",   desc: "scuro, anche se il sistema è chiaro" },
  ];

  // localStorage puo' LANCIARE, non solo restituire null: in un iframe
  // sandboxed o con la memoria del sito bloccata, il solo accedervi
  // solleva SecurityError e senza try/catch l'eccezione interrompe il
  // gestore prima che il comando venga inserito — nessun errore
  // visibile, solo un comando che non compare.
  let memoria = null;

  function leggi() {
    let t = memoria;
    try { t = localStorage.getItem("tema"); } catch (e) { /* memoria */ }
    return STATI.some(s => s.id === t) ? t : "auto";
  }

  function applica(id) {
    memoria = id === "auto" ? null : id;
    if (id === "auto") {
      delete document.documentElement.dataset.tema;
      try { localStorage.removeItem("tema"); } catch (e) { /* memoria */ }
    } else {
      document.documentElement.dataset.tema = id;
      try { localStorage.setItem("tema", id); } catch (e) { /* memoria */ }
    }
  }

  document.addEventListener("DOMContentLoaded", () => {
    const dove = document.querySelector(".sidebar-inner");
    if (!dove) return;

    let corrente = leggi();

    const gruppo = document.createElement("div");
    gruppo.className = "tema";
    gruppo.setAttribute("role", "radiogroup");
    gruppo.setAttribute("aria-label", "Tema della pagina");

    const eti = document.createElement("span");
    eti.className = "tema-eti";
    eti.textContent = "tema";
    gruppo.appendChild(eti);

    const posizioni = STATI.map(s => {
      const b = document.createElement("button");
      b.type = "button";
      b.className = "tema-p";
      b.textContent = s.voce;
      b.title = s.desc;
      b.dataset.val = s.id;
      b.setAttribute("role", "radio");
      gruppo.appendChild(b);
      return b;
    });

    function segna(id) {
      corrente = id;
      posizioni.forEach(b => {
        const attivo = b.dataset.val === id;
        b.setAttribute("aria-checked", attivo ? "true" : "false");
        // tabindex mobile: il gruppo e' UNA fermata di tabulazione,
        // e le frecce muovono fra le posizioni. E' il comportamento
        // atteso da un radiogroup, non tre bottoni in fila.
        b.tabIndex = attivo ? 0 : -1;
      });
    }

    function scegli(i, muoviFuoco) {
      const s = STATI[(i + STATI.length) % STATI.length];
      applica(s.id);
      segna(s.id);
      if (muoviFuoco) posizioni[STATI.indexOf(s)].focus();
    }

    posizioni.forEach((b, i) => {
      b.addEventListener("click", () => scegli(i, false));
      b.addEventListener("keydown", (e) => {
        const k = e.key;
        if (k === "ArrowRight" || k === "ArrowDown") { e.preventDefault(); scegli(i + 1, true); }
        else if (k === "ArrowLeft" || k === "ArrowUp") { e.preventDefault(); scegli(i - 1, true); }
        else if (k === "Home") { e.preventDefault(); scegli(0, true); }
        else if (k === "End") { e.preventDefault(); scegli(STATI.length - 1, true); }
      });
    });

    segna(corrente);

    // In alto, sotto l'intestazione. In fondo alla sidebar finiva sotto
    // la piega: 24 moduli e 6 intestazioni superano i 100vh, e il
    // comando era raggiungibile solo scorrendo tutto l'indice.
    const sub = dove.querySelector(".brand-sub");
    if (sub && sub.nextSibling) dove.insertBefore(gruppo, sub.nextSibling);
    else dove.insertBefore(gruppo, dove.firstChild);
  });
})();
