#!/usr/bin/env python3
"""
Lab 19.1 — MoE: perche' i conti dei densi non si trasferiscono.   CPU.

    python3 corso/labs/lab_19_1_moe.py

Un router vero, addestrato, in quattro configurazioni. Il termine di
bilanciamento viene misurato in entrambe le direzioni: quanto salva
quando serve, e quanto costa quando non serve.
"""

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


torch.manual_seed(0)

# ----------------------------------------------------------------------
sez("1. Parametri TOTALI contro parametri ATTIVI")

H, INTER, L = 4096, 14336, 32
CONF = [
    ("denso",              1,  1),
    ("MoE  8 esperti, top-2",  8,  2),
    ("MoE 64 esperti, top-2", 64,  2),
]
print(f"  hidden={H}, intermedio={INTER}, {L} strati\n")
print(f"  {'configurazione':<24} {'par. MLP TOTALI':>17} {'ATTIVI/token':>14} {'rapporto':>10}")
for nome, n_exp, topk in CONF:
    per_exp = 3 * H * INTER
    tot = per_exp * n_exp * L
    att = per_exp * topk * L
    print(f"  {nome:<24} {tot / 1e9:>14.1f} G {att / 1e9:>12.1f} G {tot / att:>9.1f}x")

print("""
  Due numeri diversi, e servono a due domande diverse:

    TOTALI  -> quanta MEMORIA serve. Tutti gli esperti stanno in memoria,
               anche quelli che questo token non usa.
    ATTIVI  -> quanto CALCOLO costa un token.

  Un modello "8x7B" non e' un 56B: e' un modello con ~56B di parametri
  da tenere in memoria e ~13B di calcolo per token. Chi cita un solo
  numero sta rispondendo a una domanda diversa dalla tua.

  Conseguenza diretta sull'aritmetica di M06: la voce PESI si calcola
  sui TOTALI. Il risparmio del MoE e' sul calcolo, non sulla memoria.""")


# ----------------------------------------------------------------------
sez("2. Il router: un piccolo classificatore")

class MoE(nn.Module):
    def __init__(self, d=64, n_exp=8, topk=2, d_int=128, forza=0.0):
        super().__init__()
        self.n_exp, self.topk = n_exp, topk
        self.router = nn.Linear(d, n_exp, bias=True)
        if forza:                      # sbilanciamento IMPOSTO, per lo scenario 3
            with torch.no_grad():
                self.router.bias.zero_(); self.router.bias[0] = forza
        self.esperti = nn.ModuleList(
            [nn.Sequential(nn.Linear(d, d_int), nn.GELU(), nn.Linear(d_int, d))
             for _ in range(n_exp)])

    def forward(self, x):
        logit = self.router(x)                     # [N, n_exp]
        pesi_tutti = logit.softmax(-1)
        val, idx = pesi_tutti.topk(self.topk, dim=-1)
        val = val / val.sum(-1, keepdim=True)
        out = torch.zeros_like(x)
        for k in range(self.topk):
            for e in range(self.n_exp):
                sel = idx[:, k] == e
                if sel.any():
                    out[sel] += val[sel, k:k + 1] * self.esperti[e](x[sel])
        return out, pesi_tutti, idx


print("""  Il router e' uno strato lineare da d a n_esperti, seguito da softmax
  e da un top-k. Nient'altro: non c'e' semantica, non c'e' assegnazione
  per argomento. E' un classificatore che impara insieme al resto.

  Il rischio noto: se il router impara che un esperto va bene per tutto,
  lo usera' per tutto, e gli altri non riceveranno gradiente. Nella
  prossima sezione si misura quanto costa quando succede - e si vede
  anche che in questo giocattolo NON succede spontaneamente.""")


# ----------------------------------------------------------------------
sez("3. Il termine di bilanciamento: cosa costa e cosa salva")

def addestra(bil=False, forza=0.0, coef=0.5, passi=1500, d=64, n_exp=8, seed=0):
    torch.manual_seed(seed)
    m = MoE(d=d, n_exp=n_exp, forza=forza)
    opt = torch.optim.AdamW(m.parameters(), lr=3e-3)
    g = torch.Generator().manual_seed(7)
    X = torch.randn(1024, d, generator=g)
    Wt = torch.randn(d, d, generator=g) * 0.2
    Y = torch.tanh(X @ Wt)                       # compito APPRENDIBILE
    for _ in range(passi):
        i = torch.randint(0, 1024, (64,))
        out, pesi, idx = m(X[i])
        loss = ((out - Y[i]) ** 2).mean()
        if bil:
            c = torch.zeros(n_exp)
            for k in range(m.topk):
                c += torch.bincount(idx[:, k], minlength=n_exp).float()
            loss = loss + coef * n_exp * ((c / c.sum()) * pesi.mean(0)).sum()
        loss.backward(); opt.step(); opt.zero_grad()
    with torch.no_grad():
        _, _, idx = m(X)
        c = torch.zeros(n_exp)
        for k in range(m.topk):
            c += torch.bincount(idx[:, k], minlength=n_exp).float()
        # ATTENZIONE alla normalizzazione: con top-k ogni token produce k
        # assegnazioni, quindi c.sum() == k * N, non N. f e' la QUOTA DELLE
        # ASSEGNAZIONI per esperto, non la frazione di token: un esperto al
        # 50% delle assegnazioni con top-2 puo' essere scelto dal 100% dei
        # token. Le due misure vanno tenute distinte, e le calcolo entrambe.
        f = c / c.sum()
        top = int(f.argmax())
        tok_top = (idx == top).any(dim=1).float().mean().item()
        ent = -(f * f.clamp(min=1e-9).log()).sum()
        mse = ((m(X)[0] - Y) ** 2).mean().item()
    return f, ent.item(), mse, tok_top


ENT_MAX = math.log(8)
print(f"  8 esperti, top-2, compito apprendibile. Entropia massima {ENT_MAX:.3f}.")
print("  'assegn. top' = quota delle assegnazioni top-2 andata all'esperto piu'")
print("  usato (le quote sommano a 1 su 2N assegnazioni). '% token' = frazione")
print("  di token che quell'esperto se lo trova fra i due scelti.\n")
print(f"  {'scenario':<30} {'assegn. top':>13} {'% token':>9} {'entropia':>10} {'MSE':>10}")
righe = {}
for nome, kw in [("routing libero",                {}),
                 ("+ termine di bilanciamento",    {"bil": True}),
                 ("router SBILANCIATO a forza",    {"forza": 8.0}),
                 ("sbilanciato + bilanciamento",   {"forza": 8.0, "bil": True})]:
    f, ent, mse, tok_top = addestra(**kw)
    righe[nome] = (f.max().item(), ent, mse, tok_top)
    print(f"  {nome:<30} {100 * f.max():>12.1f}% {100 * tok_top:>8.1f}% {ent:>10.3f} {mse:>10.4f}")

lib = righe["routing libero"]
bil = righe["+ termine di bilanciamento"]
sbil = righe["router SBILANCIATO a forza"]
cura = righe["sbilanciato + bilanciamento"]

print(f"""
  ONESTA' SU COSA NON E' SUCCESSO: in questo giocattolo il router NON
  collassa da solo. Con ingressi isotropi e nessuna struttura da
  specializzare, gli esperti restano bilanciati anche senza aiuto
  ({100 * lib[0]:.1f}% al piu' usato, entropia {lib[1]:.3f} su {ENT_MAX:.3f}).

  Non lo forzo fingendo che sia spontaneo: lo IMPONGO nella terza riga,
  con un bias nel router, per far vedere che aspetto hanno le metriche
  quando succede davvero.

  COME VANNO LETTE QUESTE RIGHE. Sono QUATTRO SCENARI COSTRUITI, non
  quattro misure della stessa cosa. La terza riga non e' 'il router che
  collassa': e' un router a cui ho imposto forza=8.0 sui logit e che ho
  poi RIADDESTRATO. Fra riga 1 e riga 3 non cambia solo il bilanciamento -
  cambia il problema di ottimizzazione, e con esso la traiettoria. Un
  singolo seme per riga, nessun intervallo.

  Quindi le righe mostrano CHE ASPETTO HANNO le metriche nei quattro casi.
  NON isolano lo sbilanciamento come causa del peggioramento, e non
  stimano di quanto costi. Per una conclusione causale servirebbero
  condizioni abbinate, piu' semi e intervalli -> M12.

  Le tre cose che si leggono, con quel limite in mente:

  1. SBILANCIAMENTO E MSE PEGGIORE COMPAIONO INSIEME.
     Nello scenario forzato l'esperto piu' usato prende il {100 * sbil[0]:.0f}% delle
     assegnazioni ({100 * sbil[3]:.0f}% dei token se lo trova fra i due scelti) e
     l'MSE va da {lib[2]:.4f} a {sbil[2]:.4f}: {sbil[2] / lib[2]:.1f} volte peggio. Paghi la memoria
     di 8 esperti e ne sfrutti l'equivalente di {math.exp(sbil[1]):.1f}.

  2. IL BILANCIAMENTO RECUPERA, quando c'e' qualcosa da recuperare.
     Sulla riga sbilanciata riporta l'entropia a {cura[1]:.3f} e l'MSE da
     {sbil[2]:.4f} a {cura[2]:.4f}.

  3. MA NON E' GRATIS. Sulla riga gia' bilanciata PEGGIORA l'MSE da
     {lib[2]:.4f} a {bil[2]:.4f}: {bil[2] / lib[2]:.1f} volte peggio, per bilanciare qualcosa
     che era gia' bilanciato. Anche questo e' un singolo run: dice che il
     termine ausiliario PUO' costare, non che costi sempre {bil[2] / lib[2]:.1f} volte.
     Compete con la loss principale, e il coefficiente e' un iperparametro
     con un costo da misurare, non da assumere.

  Metriche da registrare in ogni addestramento MoE:
    - quota delle assegnazioni top-k per esperto (occhio: somma su k*N,
      non su N - non e' la frazione di token)
    - frazione di token che seleziona ciascun esperto (somma a k)
    - entropia del routing, confrontata col suo massimo log(N)
    - token scartati per capacita', dove il meccanismo esiste""")


sez("4. Perche' il fine-tuning di un MoE si comporta diversamente")

print("""  1. GRADIENTE SPARSO
     Ogni token aggiorna solo gli esperti che ha scelto. Con un dataset
     di fine-tuning piccolo e specialistico, molti esperti possono non
     ricevere quasi nessun gradiente: restano al valore del pretraining
     mentre pochi altri si spostano molto.

  2. IL ROUTER PUO' SPOSTARSI
     Se il fine-tuning cambia la distribuzione degli ingressi, il router
     puo' cambiare le sue scelte - e portare token verso esperti che non
     erano stati addestrati per loro. E' una fonte di regressione che
     nei densi non esiste.

  3. LE PERCENTUALI DEI DENSI NON VALGONO
     In M04 avevi misurato le quote di parametri di un denso. In un MoE
     la quota dell'MLP esplode, perche' e' moltiplicata per il numero
     di esperti. Un LoRA sui soli target di attention tocca una frazione
     ancora piu' piccola del modello. -> M14

  4. IL CONTO DI MEMORIA CAMBIA
     La voce PESI segue i parametri TOTALI. Il conto di M06 fatto sui
     parametri "attivi" sottostima la memoria di un fattore pari al
     rapporto della sezione 1.""")


sez("5. Cosa questo lab NON dimostra")

print("""  Il modello qui e' minuscolo, i dati sono casuali, gli esperti sono
  MLP giocattolo. Il collasso e' REALE e le metriche sono quelle giuste,
  ma:

    - la soglia a cui un MoE vero collassa dipende da scala, dati e
      coefficiente del termine di bilanciamento
    - il coefficiente 0.5 usato qui non e' un valore consigliato: e'
      quello che rende visibile l'effetto in questo giocattolo
    - esistono varianti di routing e di bilanciamento diverse da quella
      implementata qui, e alcune non usano affatto un termine ausiliario

  Quello che si porta via: QUALI metriche guardare, e che il collasso
  non produce errori - solo capacita' sprecata. Il resto va misurato
  sul modello vero, con gli strumenti di M12.""")

print()
