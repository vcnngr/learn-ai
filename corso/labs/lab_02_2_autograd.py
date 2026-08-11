#!/usr/bin/env python3
"""
Lab 02.2 — Autograd: chi calcola i gradienti al posto tuo.   CPU.

    python3 corso/labs/lab_02_2_autograd.py

Verifica numericamente le formule matriciali di M02, mostra l'accumulo
del gradiente e conferma che gradiente e peso hanno la stessa forma.
"""

import torch


def sez(titolo):
    print(f"\n{'─' * 66}\n{titolo}\n{'─' * 66}")


torch.manual_seed(0)

sez("1. Il 6 che avevi calcolato a mano")

x = torch.tensor(3.0, requires_grad=True)
y = x ** 2
print(f"  x = {x.item()},  y = x^2 = {y.item()}")
print(f"  x.grad prima di backward() = {x.grad}")
y.backward()
print(f"  x.grad dopo  backward()    = {x.grad.item()}")


sez("2. Il gradiente ha SEMPRE la forma del tensore")

W = torch.randn(11008, 4096, requires_grad=True)
X = torch.randn(32, 4096)
loss = (X @ W.T).sum()
loss.backward()

print(f"  W.shape       = {list(W.shape)}")
print(f"  W.grad.shape  = {list(W.grad.shape)}   <- identica")
byte = W.numel() * W.element_size()
print(f"\n  W       = {W.numel():>12,} numeri x {W.element_size()} byte = {byte / 1e6:7.1f} MB")
print(f"  W.grad  = {W.grad.numel():>12,} numeri x {W.element_size()} byte = {byte / 1e6:7.1f} MB")
print(f"  totale                                        {2 * byte / 1e6:7.1f} MB")
print("\n  Un solo tensore di un solo strato. Ecco il primo raddoppio di M06.")


sez("3. Verifica delle formule matriciali")

N, IN, OUT = 5, 4, 3
X = torch.randn(N, IN, requires_grad=True)
W = torch.randn(IN, OUT, requires_grad=True)
Y = X @ W
G = torch.randn(N, OUT)          # gradiente arrivato dall'alto: dL/dY
Y.backward(G)

dW_atteso = X.T @ G              # [IN,N] @ [N,OUT] -> [IN,OUT]
dX_atteso = G @ W.T              # [N,OUT] @ [OUT,IN] -> [N,IN]

print(f"  Y = X @ W      X={list(X.shape)}  W={list(W.shape)}  Y={list(Y.shape)}\n")
print(f"  dL/dW = X.T @ (dL/dY)   -> shape {list(dW_atteso.shape)}"
      f"   coincide con autograd: {torch.allclose(W.grad, dW_atteso, atol=1e-5)}")
print(f"  dL/dX = (dL/dY) @ W.T   -> shape {list(dX_atteso.shape)}"
      f"   coincide con autograd: {torch.allclose(X.grad, dX_atteso, atol=1e-5)}")
print("\n  Non serve memorizzarle: le shape ricostruiscono le formule da sole.")


sez("4. Per il gradiente dei pesi serve X. Ecco le attivazioni.")

X = torch.randn(8, 512, 4096)
mb = X.numel() * X.element_size() / 1e6
print(f"  ingresso di UNO strato: {list(X.shape)} float32 = {mb:.1f} MB")
print(f"  60 strati, tutti trattenuti insieme  = {mb * 60 / 1000:.1f} GB")
print(f"  raddoppiando il batch a 16           = {mb * 60 * 2 / 1000:.1f} GB")
print("\n  Cresce linearmente con batch x lunghezza sequenza.")
print("  Il gradient checkpointing ne tiene solo alcuni e ricalcola gli altri:")
print("  meno memoria, piu' tempo. -> M07")


sez("4b. .grad esiste solo per i tensori FOGLIA")

x = torch.tensor(3.0, requires_grad=True)   # foglia: creata da te
u = x * 2                                    # NON foglia: risultato di un'operazione
y = u ** 2
y.backward()

print(f"  x = 3, u = 2x, y = u^2")
print(f"  x.grad = {x.grad}        <- c'e' (foglia)")
print(f"  u.grad = {u.grad}       <- None (non foglia), pur essendo stato calcolato")
print("\n  Il gradiente di u ESISTE durante la propagazione (serve per arrivare a x)")
print("  ma viene scartato subito: conservare tutti gli intermedi e' insostenibile.")
print("  Per trattenerlo: u.retain_grad() PRIMA di backward().")
print("\n  requires_grad non vuol dire 'cosa da imparare': vuol dire 'tracciami'.")
print("  Cosa viene imparato lo decide l'optimizer, non questo flag.")


sez("5. .grad si ACCUMULA, non si sovrascrive")

x = torch.tensor(3.0, requires_grad=True)
for i in range(1, 4):
    (x ** 2).backward()
    print(f"  backward() #{i}  ->  x.grad = {x.grad.item():5.1f}   (atteso 6 ogni volta)")

x.grad = None
(x ** 2).backward()
print(f"  dopo azzeramento ->  x.grad = {x.grad.item():5.1f}")
print("\n  Serve per la gradient accumulation (simulare batch grandi).")
print("""
  Se dimentichi optimizer.zero_grad(), l'effetto GARANTITO e' questo:
  il gradiente su cui l'optimizer fa il passo include la storia dei
  passi precedenti, quindi l'aggiornamento non e' quello che credi -
  e la norma del gradiente cresce di passo in passo.

  Quello che NON e' garantito e' un sintomo riconoscibile nella loss.
  Non esiste 'la loss SALE': puo' salire, oscillare, saturare, divergere
  in NaN, o scendere lo stesso piu' lentamente. M03 costruisce cinque
  guasti diversi e mostra che dalla curva non li distingui. Il segnale
  da guardare e' la norma del gradiente, non la loss. -> M03, M21""")


sez("6. Softmax + cross-entropy: il gradiente e' una sottrazione")

z = torch.tensor([2.0, 1.0, 0.1], requires_grad=True)
k = 2                                            # la risposta giusta e' l'indice 2
loss = torch.nn.functional.cross_entropy(z.unsqueeze(0), torch.tensor([k]))
loss.backward()

p = torch.softmax(z, dim=0)
y = torch.zeros(3)
y[k] = 1.0

print(f"  logit z            = {[round(v, 2) for v in z.tolist()]}")
print(f"  p = softmax(z)     = {[round(v, 4) for v in p.tolist()]}   (somma = {p.sum():.1f})")
print(f"  y (risposta giusta)= {[round(v, 1) for v in y.tolist()]}")
print(f"\n  p - y              = {[round(v, 4) for v in (p - y).tolist()]}")
print(f"  z.grad da autograd = {[round(v, 4) for v in z.grad.tolist()]}")
print(f"  coincidono: {torch.allclose(z.grad, p - y, atol=1e-6)}")
print("\n  'Dove hai dato probabilita' e non dovevi, abbassa. Dove dovevi e")
print("  non l'hai data, alza. In proporzione a quanto hai sbagliato.'")


sez("7. log(V): utile, ma SOLO a inizializzazione casuale")

print("  Modello appena inizializzato = nessuna preferenza = 1/V per ogni voce.")
print(f"\n  {'vocabolario':>12}  {'loss attesa':>12}")
for V in (32_000, 128_000, 256_000):
    z = torch.zeros(1, V)                        # logit perfettamente uniformi
    l = torch.nn.functional.cross_entropy(z, torch.tensor([0]))
    print(f"  {V:>12,}  {l.item():>12.2f}")

print("\n  LIMITE DI VALIDITA' - leggilo, e' il punto della sezione:")
print("  Questo vale per un modello inizializzato A CASO (M03, M09).")
print("  NON vale per fine-tuning o LoRA su un modello preaddestrato: quello")
print("  prevede gia' bene, e una loss iniziale fra 1 e 3 e' NORMALE.")
print("  Non prova alcuna contaminazione.")
print("\n  Anche a init casuale e' un valore ATTESO, non esatto: i logit reali")
print("  non sono perfettamente uniformi, e pesi legati / posizioni escluse")
print("  dalla loss lo spostano di qualcosa.")

sez("8. CONTROLLO A - parita': binario, passa o non passa")

torch.manual_seed(1)
F = torch.nn.functional
V, N, D = 5000, 64, 128

# Modello "base" competente ma NON perfetto: i target sono campionati dalla
# distribuzione del modello stesso, quindi la loss e' quella di un modello
# gia' addestrato (ben sotto log(V), ma lontana da zero).
x = torch.randn(N, D)
W_base = torch.randn(D, V) * 0.45
base_logits = x @ W_base
target = torch.multinomial(base_logits.softmax(-1), 1).squeeze(1)


def con_lora(x, W, A, B, alpha=1.0):
    """LoRA: uscita = x@W + alpha * (x@A)@B"""
    return x @ W + alpha * ((x @ A) @ B)


r = 8
A = torch.randn(D, r) * 0.02                        # init standard: A casuale...
B_neutro = torch.zeros(r, V)                        # ...B a ZERO -> contributo nullo

loss_base = F.cross_entropy(x @ W_base, target)
loss_passo0 = F.cross_entropy(con_lora(x, W_base, A, B_neutro), target)

print(f"  loss di un modello casuale = {torch.log(torch.tensor(float(V))).item():.4f}  (= log V)")
print(f"  loss modello BASE          = {loss_base.item():.10f}  <- competente, non perfetto")
print(f"  loss al PASSO ZERO         = {loss_passo0.item():.10f}")
print(f"  scarto                     = {abs(loss_base - loss_passo0).item():.2e}   <- deve essere ~0")
print("\n  Perche': con B=0 il termine (x@A)@B vale zero. Il modello CON adattatore")
print("  E' il modello base. Non e' una coincidenza: e' l'init standard di LoRA.")
print("  Scarti a 1e-6 = rumore numerico. Scarti a 0.1 = qualcosa non torna.")

print("\n  Cosa succede se l'adattatore NON e' neutro all'init:")
# B casuale, scalato perche' il contributo sia meta' dei logit base:
# cosi' il fallimento e' visibile e riproducibile, non dipende dal seed.
B_rotto = torch.randn(r, V)
delta = (x @ A) @ B_rotto
B_rotto = B_rotto * (0.5 * base_logits.std() / delta.std())
loss_rotto = F.cross_entropy(con_lora(x, W_base, A, B_rotto), target)
print(f"    loss al passo zero    = {loss_rotto.item():.4f}")
print(f"    scarto dalla base     = {(loss_rotto - loss_base).item():+.4f}   <- il controllo A FALLISCE")
print("    Nessuna colpa dei dati. E' la configurazione dell'adattatore.")
print("\n    Attenzione al SEGNO: qui e' positivo, ma non e' una legge.")
print("    Dipende da quanto e' forte la perturbazione. Misuriamolo:")
print(f"\n    {'ampiezza':>9}  {'scarto medio':>13}  {'negativi':>9}  {'segno'}")
for scala in (0.5, 0.1, 0.02, 0.005):
    scarti = []
    for s in range(40):
        g = torch.Generator().manual_seed(1000 * int(scala * 1000 + 1) + s)
        Br = torch.randn(r, V, generator=g)
        d = (x @ A) @ Br
        Br = Br * (scala * base_logits.std() / d.std())
        scarti.append((F.cross_entropy(con_lora(x, W_base, A, Br), target) - loss_base).item())
    neg = sum(1 for v in scarti if v < 0)
    medio = sum(scarti) / len(scarti)
    verdetto = "concorde" if neg == 0 or neg == len(scarti) else "TESTA O CROCE"
    print(f"    {scala:>9.3f}  {medio:>+13.4f}  {neg:>6}/40  {verdetto}")

print("\n    Letto correttamente: in QUESTO sweep, su UN batch fisso, con")
print("    perturbazione forte il segno e' risultato concorde in 40/40; con")
print("    perturbazione debole e' andato in entrambe le direzioni.")
print("    Non e' una legge e non fissa una soglia: bastano a mostrare che")
print("    il segno non e' un criterio.")
print("\n    In una diagnosi reale sei spesso nella zona bassa - se lo scarto")
print("    fosse enorme te ne saresti gia' accorto. Regola sicura, una sola:")
print("    guarda |scarto| contro la tolleranza, mai il segno.")

print("\n  Se A fallisce, una condizione che credevi identica non lo e'.")
print("  Le cause NON si distinguono guardando lo scarto: si distinguono")
print("  con un intervento che ne elimina una alla volta.")
print("  Ogni test presuppone che il precedente sia PASSATO. Saltarne uno")
print("  attribuisce la colpa allo stadio sbagliato.\n")
scala_test = [
    ("modalita' o RNG diversi",        ["forza eval() e lo STESSO seed nei due percorsi"]),
    ("adattatore non neutro all'init", ["azzera esplicitamente B e rimisura"]),
    ("preparazione dei dati",          ["confronta l'INPUT STRUTTURATO passato al template:",
                                        "lista messaggi, campi mappati, prompt di sistema"]),
    ("template diverso",               ["a parita' di input strutturato: applica il template",
                                        "in entrambi e confronta il testo prodotto"]),
    ("tokenizer diverso",              ["a parita' di testo: tokenizza QUEL testo con entrambi",
                                        "e confronta gli input_ids"]),
    ("maschera della loss diversa",    ["confronta il tensore labels ELEMENTO PER ELEMENTO",
                                        "(quali posizioni sono ignore_index, e l'allineamento)"]),
]
for i, (causa, righe) in enumerate(scala_test, 1):
    print(f"    {i}. {causa:32} -> {righe[0]}")
    for r_ in righe[1:]:
        print(f"       {'':32}    {r_}")

print("\n  Tre trappole in questi test:")
print("   * TESTO DIVERSO non vuol dire 'e' il template'. Vuol dire solo che la")
print("     divergenza e' a monte della tokenizzazione: lettura del record, campi")
print("     mappati, prompt di sistema, normalizzazione degli spazi. Per accusare")
print("     il template devi prima aver dimostrato input strutturato identico.")
print("     Altrimenti la conclusione onesta si ferma a 'problema a monte'.")
print("   * gli INPUT_IDS da soli non separano template da tokenizer: entrambi")
print("     li cambiano.")
print("   * CONTARE le posizioni mascherate non basta: due maschere possono avere")
print("     lo stesso totale e coprire posizioni diverse. Il caso peggiore e' lo")
print("     sfasamento di una posizione: stesso conteggio, allineamento errato,")
print("     e il modello impara a prevedere il token sbagliato.")
print("\n  E un intreccio: nelle librerie diffuse il chat template e' conservato")
print("  DENTRO la config del tokenizer, quindi le due ipotesi non sono")
print("  indipendenti. Applica esplicitamente lo stesso template nei due")
print("  percorsi, invece di fidarti di quello che ogni tokenizer si porta")
print("  dietro: altrimenti due errori possono compensarsi e il test 'passa'.")


sez("9. CONTROLLO B - apprendimento: rumoroso, serve statistica")

print("  Solo DOPO che A passa. Qui i confronti sono fra misure diverse.")
print("\n  La STESSA loss, sullo STESSO modello immobile, su batch diversi:")
gen = torch.Generator().manual_seed(7)
valori = []
for _ in range(8):
    idx = torch.randperm(N, generator=gen)[:16]
    valori.append(F.cross_entropy(base_logits[idx], target[idx]).item())
media = sum(valori) / len(valori)
print(f"    {'  '.join(f'{v:.3f}' for v in valori)}")
print(f"    media {media:.3f}   min {min(valori):.3f}   max {max(valori):.3f}"
      f"   ampiezza {max(valori) - min(valori):.3f}")

print("\n  Il modello non e' cambiato di un bit: quella variazione e' solo")
print("  scelta del batch. Un singolo numero, in qualsiasi direzione, non")
print("  conclude niente.")
print("\n  ATTENZIONE: il range di 8 campioni NON e' una soglia statistica.")
print("  Non e' vero che 'sotto questo range non significa niente': una")
print("  differenza piccola ma sistematica puo' essere reale, e una grande")
print("  puo' essere caso. Per dirlo servono insieme held-out, misure")
print("  appaiate e intervalli di confidenza. -> M12")

print()
