#!/usr/bin/env python3
"""
Lab 17.1 — RL online e distillazione: perche' costano cosi' tanto.   CPU.

    python3 corso/labs/lab_17_1_rl_distill.py

Il costo dell'RL online non e' un'opinione: si calcola. E la distillazione
non e' "dataset sintetico": la differenza si misura.
"""

import math
import random

import torch
import torch.nn.functional as F


def sez(t):
    print(f"\n{'─' * 72}\n{t}\n{'─' * 72}")


GB = 1e9
torch.manual_seed(0)
random.seed(0)

# ----------------------------------------------------------------------
sez("1. Quanti modelli tiene in memoria ciascun metodo")

P = 27e9
METODI = {
    "SFT":                 {"addestrato": 1, "congelati": 0, "genera": False},
    "DPO":                 {"addestrato": 1, "congelati": 1, "genera": False},
    "PPO (classico)":      {"addestrato": 2, "congelati": 2, "genera": True},
    "GRPO":                {"addestrato": 1, "congelati": 1, "genera": True},
}
NOTE = {
    "SFT":            "solo il modello",
    "DPO":            "+ riferimento congelato",
    "PPO (classico)": "policy + value addestrati; riferimento + reward congelati",
    "GRPO":           "niente value model: il baseline viene dal GRUPPO di campioni",
}

# Stessa configurazione LoRA del caso di studio usata da M06: tutti e
# sette i moduli a r=32 su un denso da 27B -> 132.644.864 addestrabili.
# La conta e' in lab_14_1_lora.py sez.1, la config in profilo.js (caso.lora).
LORA_FRAC = 0.00491

print(f"  Modello da {P:.0f} parametri, LoRA sullo {100 * LORA_FRAC:.3f}%"
      f" (caso di studio: 7 moduli a r=32), pesi bf16.\n")
print(f"  {'metodo':<18} {'addestr.':>9} {'congelati':>10} {'genera?':>8}"
      f" {'GB statici':>11}  composizione")
for nome, m in METODI.items():
    # pesi di tutti i modelli + stati optimizer solo sugli addestrabili
    pesi = (m["addestrato"] + m["congelati"]) * P * 2
    addestrabili = m["addestrato"] * P * LORA_FRAC
    stati = addestrabili * (2 + 8 + 4)
    tot = (pesi + stati) / GB
    print(f"  {nome:<18} {m['addestrato']:>9} {m['congelati']:>10}"
          f" {('si' if m['genera'] else 'no'):>8} {tot:>11.1f}  {NOTE[nome]}")

print("""
  La colonna 'genera' e' quella che cambia tutto, e non si vede nei GB:
  i metodi online devono PRODURRE testo a ogni passo, prima di poter
  calcolare la loss. Questo aggiunge:
    - una KV cache viva durante la generazione (M06: qui esiste davvero)
    - latenza sequenziale: un token alla volta, non parallelizzabile
    - un secondo profilo di memoria, diverso da quello di addestramento""")


# ----------------------------------------------------------------------
sez("2. Il costo nascosto: la generazione e' sequenziale")

print("  Un passo di addestramento supervisionato elabora TUTTA la")
print("  sequenza in parallelo. Un passo online deve prima generarla,")
print("  un token alla volta.\n")
print(f"  {'token da generare':>18} {'passaggi sequenziali':>22} {'vs 1 forward':>14}")
for n in (1, 64, 256, 1024):
    print(f"  {n:>18} {n:>22} {n:>13}x")

print("""
  Non e' un fattore costante da moltiplicare al costo: e' un'operazione
  di natura DIVERSA, che non si parallelizza sulla lunghezza. Con GRPO,
  che campiona piu' risposte per prompt, si moltiplica ancora per il
  numero di campioni del gruppo.

  Questo e' il motivo per cui l'RL online e' il metodo piu' costoso del
  corso, e perche' su poche GPU se ne vede la MECCANICA ma non il regime
  in cui diventa interessante.""")


# ----------------------------------------------------------------------
sez("3. GRPO: da dove viene il baseline")

print("""  In PPO un secondo modello (il 'value model') stima quanto e' buono
  uno stato, per sapere se una ricompensa e' meglio o peggio del previsto.
  Costa un modello intero da addestrare.

  GRPO toglie quel modello e usa il GRUPPO: campiona K risposte per lo
  stesso prompt e usa la loro media come riferimento.""")

def ricompense(k, seed):
    r = random.Random(seed)
    return [r.gauss(0.5, 0.25) for _ in range(k)]

print(f"\n  {'K campioni':>11} {'media':>8} {'errore std':>11} {'vs K/4':>8}")
prec = None
for k in (2, 4, 8, 16, 64):
    tot_m, tot_s = 0.0, 0.0
    for s in range(200):
        r = ricompense(k, s)
        m = sum(r) / k
        tot_m += m
        tot_s += (sum((x - m) ** 2 for x in r) / max(1, k - 1)) ** 0.5
    err = tot_s / 200 / math.sqrt(k)
    rap = f"{prec / err:.2f}x" if prec and k >= 16 else "-"
    print(f"  {k:>11} {tot_m / 200:>8.4f} {err:>11.4f} {rap:>8}")
    if k in (4, 16):
        prec = err

print("""
  L'errore del baseline scende come 1/sqrt(K): quadruplicare i campioni
  lo dimezza. Misurato: 4->16 da' 1.83x, 16->64 da' 1.98x - il primo
  scarto viene dalla stima della deviazione, inaffidabile con K piccolo.
  E ogni campione e' una generazione completa.

  E' il baratto centrale di GRPO: niente value model da addestrare, ma
  il costo si sposta sulla generazione, che paghi K volte per prompt.
  Con K piccolo il baseline e' rumoroso; con K grande costa.""")


# ----------------------------------------------------------------------
sez("4. Distillazione: NON e' 'dataset sintetico'")

print("""  Si sente spesso chiamare 'distillazione' il fatto di generare dati
  con un modello grande e fare SFT su quelli. E' una cosa diversa, e
  la differenza si vede nell'obiettivo.""")

V = 8
logit_teacher = torch.tensor([3.0, 2.5, 1.0, 0.2, 0.0, -0.5, -1.0, -2.0])
p_teacher = logit_teacher.softmax(0)
bersaglio_secco = torch.zeros(V); bersaglio_secco[0] = 1.0

print(f"\n  Distribuzione del TEACHER su {V} token possibili:")
print("   " + "  ".join(f"{p:.3f}" for p in p_teacher.tolist()))
print(f"\n  Bersaglio SECCO (solo il token campionato):")
print("   " + "  ".join(f"{p:.3f}" for p in bersaglio_secco.tolist()))

ent = -(p_teacher * p_teacher.log()).sum()
print(f"""
  Entropia della distribuzione del teacher: {ent:.4f} nat

  Il teacher non dice solo 'la risposta e' il token 0': dice che il
  token 1 e' quasi altrettanto buono, e che il token 7 e' da escludere.
  Quella e' informazione, e il bersaglio secco la butta via tutta.""")

# quanto costa in termini di segnale
logit_student = torch.zeros(V, requires_grad=True)
kl = F.kl_div(logit_student.log_softmax(0), p_teacher, reduction="sum")
ce = F.cross_entropy(logit_student.unsqueeze(0), torch.tensor([0]))
print(f"  student uniforme -> KL dal teacher: {kl.item():.4f}"
      f"   cross-entropy sul bersaglio secco: {ce.item():.4f}")

print("""
  DISTILLAZIONE (logit-level): lo student impara a riprodurre l'intera
  DISTRIBUZIONE del teacher. Serve accesso ai logit, quindi al modello,
  e i due devono condividere il tokenizer.

  DATASET SINTETICO (sequence-level): il teacher genera testo, lo
  student fa SFT su quel testo. Serve solo l'output. Funziona anche
  con tokenizer diversi e con un teacher accessibile solo via API.

  Sono due metodi diversi con requisiti diversi. Chiamarli entrambi
  'distillazione' fa scegliere quello sbagliato.""")


# ----------------------------------------------------------------------
sez("5. La temperatura nella distillazione")

print("  Alzare la temperatura del teacher rende visibili le preferenze")
print("  sui token meno probabili - quelli dove sta l'informazione in piu'.\n")
print(f"  {'T':>5}  distribuzione del teacher")
for T in (0.5, 1.0, 2.0, 4.0):
    p = (logit_teacher / T).softmax(0)
    ent_T = -(p * p.clamp(min=1e-9).log()).sum()
    print(f"  {T:>5}  " + " ".join(f"{x:.3f}" for x in p.tolist()) + f"   entropia {ent_T:.3f}")

print("""
  A T bassa la distribuzione collassa sul massimo e la distillazione
  degenera nel bersaglio secco. A T alta si appiattisce e il segnale
  utile si confonde col rumore delle code.

  Il valore giusto dipende dal teacher e dal compito: e' un iperparametro
  da misurare (M12), non da copiare.""")


# ----------------------------------------------------------------------
sez("6. Quando ciascun metodo e' la scelta sbagliata")

print("""  RL ONLINE (GRPO, PPO)
    sbagliato se: sai gia' scrivere o preferire le risposte giuste.
                  Stai pagando la generazione a ogni passo per un
                  segnale che avresti gratis con SFT o DPO.
    sbagliato se: non hai una funzione di ricompensa affidabile. Il
                  modello ottimizzera' ESATTAMENTE quella che hai
                  scritto, comprese le sue scorciatoie.
    giusto se:    la qualita' e' verificabile automaticamente (un test
                  che passa, un risultato numerico corretto) ma non
                  sai scrivere in anticipo la risposta.

  DISTILLAZIONE
    sbagliato se: il teacher non e' migliore dello student sul TUO
                  compito. Distillare da un modello mediocre produce
                  uno student mediocre, in modo molto efficiente.
    sbagliato se: i tokenizer differiscono e volevi il livello logit.
    giusto se:    hai un modello grande che funziona e ti serve la
                  stessa capacita' a costo di inferenza minore.

  In entrambi i casi la domanda preliminare e' la stessa di M13:
  il modello NON SA, o NON SI COMPORTA? E la risposta si misura.""")

print()
