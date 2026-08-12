// ============================================================
//  tema.js — chiaro / scuro / automatico.
//
//  Tre stati, non due. "Automatico" non e' un riempitivo: e' il
//  default giusto, perche' segue la preferenza che il lettore ha
//  gia' espresso al sistema operativo. Gli altri due esistono per
//  quando quella preferenza e' sbagliata nel contesto — una stanza
//  buia, un proiettore, una schermata condivisa.
//
//  La scelta si applica in <head>, PRIMA del primo paint, con lo
//  snippet inline che ogni pagina porta. Se stesse qui, la pagina
//  comparirebbe chiara e poi diventerebbe scura: un lampo bianco a
//  ogni navigazione, su un sito che si legge di notte.
// ============================================================

(function () {
  const STATI = [
    { id: "auto",   etichetta: "auto",   titolo: "segue il sistema" },
    { id: "chiaro", etichetta: "chiaro", titolo: "chiaro, anche se il sistema è scuro" },
    { id: "scuro",  etichetta: "scuro",  titolo: "scuro, anche se il sistema è chiaro" },
  ];

  // localStorage puo' LANCIARE, non solo restituire null: in un iframe
  // sandboxed, in un'origine opaca, o con la memoria del sito bloccata,
  // il solo accedervi solleva SecurityError. Senza questi try/catch
  // l'eccezione interrompeva il gestore di DOMContentLoaded PRIMA che
  // il comando venisse aggiunto: niente eccezione visibile all'utente,
  // solo un bottone che non compare.
  //
  // La ricaduta e' su una preferenza, non su un contenuto: se non si
  // puo' ricordare, si tiene in memoria per la sessione e il corso si
  // legge lo stesso.
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

  function disegna(bottone, id) {
    const s = STATI.find(x => x.id === id);
    bottone.textContent = s.etichetta;
    bottone.title = "Tema: " + s.titolo + ". Clic per cambiare.";
    bottone.setAttribute("aria-label", "Tema: " + s.etichetta);
  }

  document.addEventListener("DOMContentLoaded", () => {
    const dove = document.querySelector(".sidebar-inner");
    if (!dove) return;

    const box = document.createElement("div");
    box.className = "tema-box";

    const etichetta = document.createElement("span");
    etichetta.className = "tema-eti";
    etichetta.textContent = "tema";

    const bottone = document.createElement("button");
    bottone.className = "tema-btn";
    bottone.type = "button";

    let corrente = leggi();
    disegna(bottone, corrente);

    bottone.addEventListener("click", () => {
      const i = STATI.findIndex(s => s.id === corrente);
      corrente = STATI[(i + 1) % STATI.length].id;
      applica(corrente);
      disegna(bottone, corrente);
    });

    box.append(etichetta, bottone);

    // IN ALTO, non in fondo.
    //
    // Prima lo appendevo alla fine di .sidebar-inner "perche' e' un
    // comando, non una voce di navigazione". Ragionamento pulito e
    // sbagliato: la sidebar ha height 100vh con overflow-y auto, e
    // sopra ci stanno 24 moduli piu' 6 intestazioni di parte. Il
    // comando finiva SOTTO LA PIEGA, raggiungibile solo scorrendo
    // l'indice fino in fondo — cioe' invisibile a chi non lo cercava.
    //
    // Sta sotto l'intestazione, dove si vede senza scorrere niente.
    const sub = dove.querySelector(".brand-sub");
    if (sub && sub.nextSibling) dove.insertBefore(box, sub.nextSibling);
    else dove.appendChild(box);
  });
})();
