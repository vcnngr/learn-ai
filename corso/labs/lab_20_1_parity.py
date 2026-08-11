#!/usr/bin/env python3
"""
Lab 20.1 — Parity: modelli sani che sembrano rotti.   CPU.

    python3 corso/labs/lab_20_1_parity.py

Cinque disallineamenti fra addestramento e inferenza, costruiti uno per
uno e misurati. Nessuno di essi produce un errore.
"""

import math

import torch
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


torch.manual_seed(0)

# ----------------------------------------------------------------------
sez("1. La scala progressiva: dove si rompe la catena")

print("""  Fra il modello che hai addestrato e quello che serve ci sono cinque
  trasformazioni. Ognuna puo' cambiare il comportamento, e nessuna
  produce un errore se sbagliata.

      1. base                    il punto di partenza
      2. base + adattatore vivo  come durante l'addestramento
      3. merged                  adattatore fuso nei pesi
      4. converted               altro formato / altro motore
      5. quantized               precisione ridotta per servire

  La regola operativa: NON confrontare il passo 1 col passo 5. Confronta
  ogni passo con il PRECEDENTE. Il primo che si discosta e' il colpevole,
  e cosi' hai isolato il problema invece di averne uno solo grande.""")


# ----------------------------------------------------------------------
sez("2. Disallineamento 1: il template")

def rendi(messaggi, template):
    corpo, coda = template
    return "".join(corpo.format(**m) for m in messaggi) + coda


T_TRAIN = ("<|im_start|>{role}\n{content}<|im_end|>\n", "<|im_start|>assistant\n")
T_SERVE = ("<|im_start|>{role}\n{content}<|im_end|>\n", "<|im_start|>assistant\n\n")

M = [{"role": "user", "content": "Ciao"}]
a, b = rendi(M, T_TRAIN), rendi(M, T_SERVE)
print(f"  addestramento: {a!r}")
print(f"  servizio:      {b!r}")
print(f"  identici? {a == b}   differenza: {len(b) - len(a)} carattere")
print(f"  primo indice diverso: {next((i for i in range(min(len(a), len(b))) if a[i] != b[i]), min(len(a),len(b)))}")
print("""
  Un solo a-capo in piu' nel prompt di generazione. A occhio i due testi
  sono identici, e nessun controllo automatico se ne accorge.

  Come si trova: confronta i due testi con repr(), carattere per
  carattere - non stampandoli, che nasconde gli spazi e gli a-capo.""")


# ----------------------------------------------------------------------
sez("3. Disallineamento 2: i token speciali aggiunti due volte")

class Tok:
    def __init__(self, bos=1000):
        self.bos = bos
    def encode(self, testo, aggiungi_speciali=True):
        ids = [ord(c) for c in testo]
        return ([self.bos] + ids) if aggiungi_speciali else ids

tok = Tok()
testo_gia_con_bos = "<BOS>Ciao"          # il template ha gia' messo il marcatore
print(f"  {'percorso':<44} {'primi id':>22}")
print(f"  {'template + encode(add_special_tokens=True)':<44}"
      f" {str(tok.encode('Ciao'))[:22]:>22}")
print(f"  {'template + encode(add_special_tokens=False)':<44}"
      f" {str(tok.encode('Ciao', False))[:22]:>22}")
print("""
  Se il chat template inserisce gia' il token di inizio e poi il
  tokenizer lo aggiunge di nuovo, la sequenza ha DUE marcatori.
  Il modello non ne ha mai visto uno doppio in addestramento.

  Come si trova: stampa i primi 5 id di una sequenza reale, sia nel
  percorso di addestramento sia in quello di servizio. Devono coincidere.""")


# ----------------------------------------------------------------------
sez("4. Disallineamento 3: il campionamento")

logit = torch.tensor([3.0, 2.8, 1.0, 0.5, 0.0, -1.0])
print("  Stesso modello, stessi logit. Cambia solo COME si sceglie il token.\n")
print(f"  {'configurazione':<34} {'p(token 0)':>12} {'p(token 1)':>12}")
for nome, T in [("temperatura 0 (greedy)", 0.0),
                ("temperatura 0.7", 0.7),
                ("temperatura 1.0", 1.0),
                ("temperatura 1.5", 1.5)]:
    if T == 0:
        p = torch.zeros_like(logit); p[logit.argmax()] = 1.0
    else:
        p = (logit / T).softmax(0)
    print(f"  {nome:<34} {p[0]:>12.4f} {p[1]:>12.4f}")

print("""
  A temperatura 0 il token 1 non esce MAI. A temperatura 1.0 esce nel
  39% dei casi. Se valuti in greedy e servi a temperatura 1, stai
  misurando un modello e servendone un altro.

  E' il disallineamento piu' banale e uno dei piu' frequenti: i default
  di generazione del motore di inferenza raramente coincidono con quelli
  che hai usato per valutare.""")


# ----------------------------------------------------------------------
sez("5. Disallineamento 4: il merge in precisione ridotta")

R, IN, OUT, ALPHA = 8, 256, 512, 16
W = torch.randn(OUT, IN) * 0.05
A = torch.randn(R, IN) * 0.02
B = torch.randn(OUT, R) * 0.02
x = torch.randn(64, IN)
scala = ALPHA / R

y_vivo = x @ W.T + scala * ((x @ A.T) @ B.T)

print(f"  {'come viene fatto il merge':<40} {'max|diff| dal vivo':>20}")
for nome, dt in [("in float32 (come l'addestramento)", torch.float32),
                 ("in bfloat16", torch.bfloat16),
                 ("in float16", torch.float16)]:
    Wm = (W.to(dt) + (scala * (B @ A)).to(dt)).to(torch.float32)
    d = (x @ Wm.T - y_vivo).abs().max().item()
    print(f"  {nome:<40} {d:>20.3e}")

print("""
  Fondere in precisione ridotta introduce un errore che il modello con
  adattatore vivo non aveva. Su un singolo strato e' piccolo; su decine
  di strati si accumula, e in M05 hai visto come.

  Regola: fondi in fp32, poi eventualmente converti i pesi fusi. Non
  fondere direttamente nel dtype di servizio.""")


# ----------------------------------------------------------------------
sez("6. Disallineamento 5: la quantizzazione post-addestramento")

def quant(t, bit=8, blocco=64):
    f = t.flatten()
    pad = (-len(f)) % blocco
    f = torch.cat([f, torch.zeros(pad)])
    b = f.view(-1, blocco)
    sc = b.abs().amax(1, keepdim=True).clamp(min=1e-8)
    liv = 2 ** (bit - 1) - 1
    q = (b / sc * liv).round().clamp(-liv, liv)
    return ((q / liv) * sc).flatten()[:t.numel()].view(t.shape)


Wm = W + scala * (B @ A)
print(f"  {'precisione di servizio':<28} {'errore sui pesi':>17} {'errore sull uscita':>20}")
for bit in (8, 4):
    Wq = quant(Wm, bit)
    ep = ((Wq - Wm).norm() / Wm.norm()).item()
    eu = ((x @ Wq.T - y_vivo).norm() / y_vivo.norm()).item()
    print(f"  {str(bit) + ' bit':<28} {ep:>17.4f} {eu:>20.4f}")

print("""
  L'errore sui pesi si trasferisce all'uscita quasi intatto. E' lo stesso
  fenomeno di M14, qui applicato DOPO l'addestramento: il modello che
  servi non e' quello che hai valutato.

  Se hai valutato il modello fuso in bf16 e ne servi uno a 4 bit, la
  differenza di qualita' che osservi puo' venire interamente da qui -
  non dal fine-tuning.""")


# ----------------------------------------------------------------------
sez("7. Il protocollo di parity")

print("""  Per ogni passo della scala, sullo STESSO insieme di prompt:

    1. calcola la loss (o la metrica) al passo N
    2. calcola la stessa al passo N+1
    3. confronta. Se lo scarto supera la tolleranza, ti fermi QUI.

  Tolleranze indicative, da tarare sul proprio caso:

    base -> adattatore all'init     scarto ~ 0        (M14: B e' zero)
    adattatore vivo -> merged       rumore numerico   (~1e-6 in fp32)
    merged -> converted             rumore numerico
    converted -> quantized          NON trascurabile, e' il prezzo scelto

  Le prime tre righe DEVONO essere quasi nulle. Se non lo sono, hai
  trovato un bug, non una perdita fisiologica.

  L'ultima riga e' l'unica dove uno scarto e' atteso - ma va MISURATO
  e accettato consapevolmente, non scoperto dopo il rilascio.""")


sez("8. Perche' questo modulo viene PRIMA della diagnosi")

print("""  Il disallineamento fra addestramento e inferenza e' una delle ipotesi
  candidate per un fine-tuning che rende poco. Ed e' anche una delle
  piu' economiche da escludere: nessun addestramento, solo confronti.

  Se non lo escludi PRIMA, rischi di passare settimane a cercare un
  difetto nei dati mentre il problema e' un a-capo nel template di
  servizio, o una temperatura di default diversa da quella con cui hai
  valutato.

  Ordine per rapporto informazione/costo:
    1. parity (questo modulo)      ore, nessun addestramento
    2. template e maschera (M10)   ore, nessun addestramento
    3. valutazione (M12)           giorni, nessun addestramento
    4. dati e iperparametri        settimane, molti addestramenti

  E' esattamente il criterio del modulo finale.""")

print()
