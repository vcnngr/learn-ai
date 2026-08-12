// Indice unico dei moduli. Le ore sono stime di lavoro reale, non di lettura.
// L'offset (ore cumulate) viene calcolato, non scritto a mano.

const MODULI = [
  { sez: "Parte I — Fondamenta" },
  { id: "m00", tag: "M00", t: "Come si usa questo corso",        f: "m00-patto.html",       h: 0.5, ok: 1, nucleo: 1 },
  { id: "m00b", tag: "M00b", t: "Mettere in piedi l'ambiente",   f: "m00b-ambiente.html",   h: 1,   ok: 1, nucleo: 1 },
  { id: "m01", tag: "M01", t: "Tensori: il mattone",             f: "m01-tensori.html",     h: 3,   ok: 1, nucleo: 1 },
  { id: "m02", tag: "M02", t: "Derivate e gradienti",            f: "m02-gradienti.html",   h: 4,   ok: 1, nucleo: 1 },
  { id: "m03", tag: "M03", t: "Cosa vuol dire addestrare",       f: "m03-addestrare.html",  h: 4,   ok: 1 },

  { sez: "Parte II — Il modello sul disco" },
  { id: "m04", tag: "M04", t: "Anatomia dei pesi",               f: "m04-anatomia.html",    h: 5,   ok: 1, nucleo: 1 },
  { id: "m05", tag: "M05", t: "Byte, dtype e precisione",        f: "m05-dtype.html",       h: 5,   ok: 1 },
  { id: "m06", tag: "M06", t: "Aritmetica della memoria",        f: "m06-memoria.html",     h: 8,   ok: 1, nucleo: 1 },

  { sez: "Parte III — Come impara" },
  { id: "m07", tag: "M07", t: "Forward, backward, autograd",     f: "m07-autograd.html",    h: 8,   ok: 1 },
  { id: "m08", tag: "M08", t: "Optimizer, learning rate, batch", f: "m08-optimizer.html",   h: 7,   ok: 1 },
  { id: "m09", tag: "M09", t: "Architettura del transformer",    f: "m09-transformer.html", h: 12,  ok: 1 },

  { sez: "Parte IV — I dati" },
  { id: "m10", tag: "M10", t: "Tokenizer e chat template",       f: "m10-tokenizer.html",   h: 6,   ok: 1, nucleo: 1 },
  { id: "m11", tag: "M11", t: "Dataset e loss masking",          f: "m11-dataset.html",     h: 7,   ok: 1, nucleo: 1 },
  { id: "m12", tag: "M12", t: "Valutazione",                     f: "m12-eval.html",        h: 8,   ok: 1, nucleo: 1 },

  { sez: "Parte V — Addestrare" },
  { id: "m13", tag: "M13", t: "Continued pretraining vs SFT",    f: "m13-cpt-sft.html",     h: 4,   ok: 1 },
  { id: "m14", tag: "M14", t: "LoRA, QLoRA, DoRA",               f: "m14-lora.html",        h: 9,   ok: 1, nucleo: 1 },
  { id: "m15", tag: "M15", t: "Fine-tuning riproducibile",       f: "m15-finetuning.html",  h: 10,  ok: 1, nucleo: 1 },
  { id: "m16", tag: "M16", t: "Preferenze: DPO e ORPO",          f: "m16-dpo.html",         h: 6.5, ok: 1 },
  { id: "m17", tag: "M17", t: "RL online e distillazione",       f: "m17-rl.html",          h: 6,   ok: 1 },

  { sez: "Parte VI — Scala e diagnosi" },
  { id: "m18", tag: "M18", t: "Parallelismo su più GPU",        f: "m18-parallelismo.html", h: 10,  ok: 1 },
  { id: "m18b", tag: "M18b", t: "Il tuo hardware: quattro regimi", f: "m18b-regimi.html", h: 8, ok: 1, nucleo: 1 },
  { id: "m19", tag: "M19", t: "MoE (opzionale)",                 f: "m19-moe.html",          h: 5,   ok: 1 },
  { id: "m20", tag: "M20", t: "Parity training / inference",     f: "m20-parity.html",       h: 7,   ok: 1, nucleo: 1 },
  { id: "m21", tag: "M21", t: "Diagnosi di un fine-tuning deludente",           f: "m21-diagnosi.html",     h: 13,  ok: 1, nucleo: 1 },
];

const ORE = MODULI.filter(m => m.id);
const TOT = ORE.reduce((s, m) => s + m.h, 0);

// Percorso ridotto "diagnostico": i moduli marcati nucleo.
// Il totale e' CALCOLATO, non scritto a mano: se cambio le ore di un
// modulo, la cifra sulla pagina segue. Un numero scritto a mano diverge
// alla prima modifica, ed e' il difetto che questo corso insegna a non
// lasciar passare.
const NUCLEO = ORE.filter(m => m.nucleo);
const TOT_NUCLEO = NUCLEO.reduce((s, m) => s + m.h, 0);
const MAXH = Math.max(...ORE.map(m => m.h));

(function computeOffsets() {
  let acc = 0;
  for (const m of MODULI) { if (!m.id) continue; m.off = acc; acc += m.h; }
})();

const fmt = h => (h % 1 === 0 ? h + "h" : h.toFixed(1) + "h");

function renderNav(current) {
  const box = document.getElementById("nav");
  if (!box) return;
  box.innerHTML = MODULI.map(m => {
    if (m.sez) return `<h4>${m.sez}</h4>`;
    const cls = ["nav", m.id === current ? "current" : "", m.ok ? "" : "todo"].join(" ").trim();
    return m.ok
      ? `<a class="${cls}" href="${m.f}"><span class="tag">${m.tag}</span><span>${m.t}</span></a>`
      : `<span class="${cls}"><span class="tag">${m.tag}</span><span>${m.t}</span></span>`;
  }).join("");
}

function renderAlloc() {
  const box = document.getElementById("alloc");
  if (!box) return;
  let html = `<div class="alloc-head"><span>offset · modulo</span><span>durata</span></div>`;
  MODULI.forEach((m, i) => {
    if (m.sez) { html += `<div class="alloc-sec">${m.sez}</div>`; return; }
    const w = Math.max(9, Math.round((m.h / MAXH) * 100));
    const tag = m.ok ? "a" : "span";
    const href = m.ok ? ` href="${m.f}"` : "";
    html += `<${tag} class="alloc-row ${m.ok ? "" : "todo"}"${href}>
      <span class="off">${fmt(m.off)}</span>
      <span class="id">${m.tag}</span>
      <span><span class="name">${m.t}${m.nucleo ? '<span class="nuc" title="nel percorso diagnostico">NUCLEO</span>' : ''}</span><span class="bar"><i data-w="${w}" data-ritardo="${i * 22}"></i></span></span>
      <span class="size">${fmt(m.h)}</span>
    </${tag}>`;
  });
  html += `<div class="alloc-total"><span>${ORE.length} moduli · ${ORE.filter(m => m.ok).length} scritti</span><span>${fmt(TOT)} totali</span></div>`;
  box.innerHTML = html;

  // Larghezza e ritardo sono valori CALCOLATI, quindi non possono stare
  // in una classe. Ma nemmeno in uno style="..." dentro innerHTML: la
  // CSP del sito ha `style-src 'self'`, e li' quello e' markup da
  // analizzare — verrebbe bloccato, e le barre resterebbero a zero.
  //
  // Impostarli via CSSOM dopo l'inserimento e' un'altra cosa: non passa
  // dal parser HTML, e la CSP non lo tocca.
  box.querySelectorAll(".bar i").forEach(el => {
    el.style.width = el.dataset.w + "%";
    el.style.animationDelay = el.dataset.ritardo + "ms";
  });
}

function renderPercorsi() {
  const box = document.getElementById("percorsi");
  if (!box) return;
  const tag = NUCLEO.map(m => m.tag).join(", ");
  const fuori = ORE.filter(m => !m.nucleo).map(m => m.tag).join(", ");
  box.innerHTML = `
    <div class="percorso">
      <span class="nome">Completo</span><span class="ore">${fmt(TOT)}</span>
      <span class="che">Tutti i ${ORE.length} moduli, nell'ordine. Alla fine sai
      scrivere un transformer da zero, contare la memoria prima di lanciare, e
      diagnosticare un fine-tuning con metodo.</span>
    </div>
    <div class="percorso ridotto">
      <span class="nome">Diagnostico</span><span class="ore">${fmt(TOT_NUCLEO)}</span>
      <span class="che"><strong>${tag}.</strong> Basta a diagnosticare un
      fine-tuning che rende meno del previsto. Non basta a costruire un modello:
      restano fuori ${fuori}, cioè tutto ciò che serve a <em>costruire</em>
      invece che a <em>capire cosa e' andato storto</em>.</span>
    </div>`;
}

function renderPageNav(current) {
  const box = document.getElementById("pagenav");
  if (!box) return;
  const flat = MODULI.filter(m => m.id && m.ok);
  const i = flat.findIndex(m => m.id === current);
  if (i < 0) return;
  const p = flat[i - 1], n = flat[i + 1];
  box.innerHTML =
    (p ? `<a href="${p.f}"><span class="dir">← precedente</span>${p.t}</a>` : `<span></span>`) +
    (n ? `<a href="${n.f}" class="pn-destra"><span class="dir">successivo →</span>${n.t}</a>` : `<span></span>`);
}

document.addEventListener("DOMContentLoaded", () => {
  const id = document.body.dataset.module || "index";
  renderNav(id); renderAlloc(); renderPercorsi(); renderPageNav(id);
});
