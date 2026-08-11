# Findings — revisore A, correttezza tecnica

Ambiente della verifica: Python 3, `torch 2.2.2`, `numpy 1.26.4`, CPU. Eseguiti tutti i 24 script: 24 terminano correttamente nel percorso disponibile. Le sezioni CUDA di M07 e M18 si fermano come previsto; risultati GPU **non verificabili**. Anche i rami `enable_gqa` e `F.rms_norm`, assenti in `torch 2.2.2`, sono **non verificabili** su versioni più recenti.

### Il profilo non parametrizza i conti del corso
**File:** `assets/profilo.js`, `index.html`, tutte le pagine `m*.html` · **Sezione:** profilo e numeri derivati  
**Severità:** BLOCKER  
**Cosa è sbagliato:** `index.html` promette che modificando hardware, modello e caso «i conti delle pagine seguono quei valori». Nel DOM, però, tutte le 23 pagine usano soltanto `data-profilo="hw"`. I valori di modello, memoria, forme, numero di GPU e caso restano hard-coded. `profilo.js` non ricalcola nessuna tabella. Cambiare, per esempio, `parametri` da 27B lascia invariati 54 GB, 432 GB e i conteggi LoRA. La procedura dichiarata non può produrre il risultato promesso.  
**Correzione concreta:** collegare ogni valore derivato ai campi `PROFILO` e calcolarlo nel rendering, inclusi tabelle e testi; in alternativa eliminare la promessa di parametrizzazione e dichiarare il profilo come sola intestazione hardware. Collegare allo stesso profilo anche gli argomenti dei lab usati per quei conti.

### AdamW a 8 byte per parametro non è una proprietà universale
**File:** `m06-memoria.html`, `m08-optimizer.html`, `labs/lab_08_1_optimizer.py`, `labs/lab_18_1_parallelismo.py` · **Sezione:** stati optimizer e memoria statica  
**Severità:** BLOCKER  
**Cosa è sbagliato:** le pagine presentano come configurazione normale pesi bf16 e due momenti AdamW fp32, quindi 8 byte per parametro e 4× il costo dei pesi. Il lab misura invece soltanto parametri fp32. Con l'AdamW nativo di `torch 2.2.2`, una prova diretta su un parametro bf16 crea `exp_avg` ed `exp_avg_sq` in bf16: 4 byte complessivi per parametro, non 8; non compare automaticamente una copia master fp32. Altri stack di mixed precision, optimizer fused/8-bit e versioni possono comportarsi diversamente. Di conseguenza 432 GB e tutte le tabelle derivate descrivono una configurazione possibile, non AdamW in generale.  
**Correzione concreta:** dichiarare esplicitamente lo stack e i dtype assunti per ogni stato; modificare il lab perché costruisca proprio quella configurazione e conti `state_dict()` e copie master reali. Se lo stack non è fissato, rendere `byte_stati` e `byte_master` parametri misurati, non costanti.

### Il lab GPU di M07 non misura il k promesso e fallisce sul checkpointing dichiarato
**File:** `m07-autograd.html`, `labs/lab_07_1_backward.py` · **Sezione:** 4–5, gradient checkpointing e k effettivo  
**Severità:** BLOCKER  
**Cosa è sbagliato:** nel ramo CUDA il codice usa `torch.utils.checkpoint.checkpoint` senza importare `torch.utils.checkpoint`; in un processo pulito con `torch 2.2.2` l'attributo non esiste e il percorso con checkpoint termina con `AttributeError`. Inoltre modello e input sono fp32, mentre la sezione stampa «bf16» e divide il picco per 2 byte. Infine calcola un solo rapporto dal picco totale, che include statico, e si limita a dire al lettore di rifare la misura a più batch: non esegue i tre punti né calcola/verifica la pendenza richiesta dalla pagina. Il ramo GPU non è stato eseguito: eventuali altri problemi sono non verificabili.  
**Correzione concreta:** importare esplicitamente il modulo checkpoint; allineare dtype reale e denominatore; eseguire nel lab almeno tre batch validi, misurare i picchi, verificare la linearità e ricavare k dalla pendenza dopo avere fissato sequenza, architettura e stack. Non stampare un «k effettivo» dal rapporto contaminato.

### Il test GPU di M18 valida il totale sbagliando la decomposizione
**File:** `m18-parallelismo.html`, `labs/lab_18_1_parallelismo.py` · **Sezione:** 5, misura DDP  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il modello del ramo CUDA è fp32 e usa AdamW nativo; la previsione confrontata col picco assume invece pesi bf16, gradienti bf16, momenti fp32 e master fp32. Entrambe le decomposizioni sommano casualmente a 16 byte per parametro, quindi un accordo del totale non convalida le quattro voci. Il lab misura solo il tempo totale per step: dalla sola efficienza 1→N non può misurare «quanta parte del tempo è comunicazione», perché include anche variazioni del calcolo, sincronizzazioni e overhead. Il ramo GPU resta non verificabile in questa revisione.  
**Correzione concreta:** costruire e ispezionare la configurazione dtype realmente prevista, riportando byte di parametri, gradienti e stato optimizer separatamente. Chiamare il risultato temporale «overhead complessivo di scaling»; per attribuirlo alla comunicazione serve una misura/profilazione separata.

### La spiegazione del loss scaling confonde underflow e overflow
**File:** `m05-dtype.html` · **Sezione:** «Perché bf16 ha vinto nel training»  
**Severità:** BLOCKER  
**Cosa è sbagliato:** il testo introduce gradienti che crescono oltre 65.504 e subito dopo presenta il loss scaling come rimedio. Moltiplicare la loss aumenta i gradienti e peggiora l'overflow. Il loss scaling serve principalmente a portare gradienti piccoli fuori dall'underflow fp16; dopo il backward vengono riportati alla scala originale. Lo scaler dinamico riduce la scala quando rileva overflow.  
**Correzione concreta:** separare i due guasti: range fp16 ridotto → rischio sia di underflow sia di overflow; loss scaling contrasta l'underflow, mentre il controllo dinamico della scala reagisce agli overflow. Dire che bf16 spesso evita questa necessità grazie all'esponente più ampio, con comportamento dipendente dallo stack numerico.

### Il clipping non tiene vicino a 1 il prodotto delle Jacobiane
**File:** `m02-gradienti.html` · **Sezione:** «La regola della catena»  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il testo raggruppa normalizzazione, residui e gradient clipping come meccanismi che tengono il prodotto della catena vicino a 1. Il clipping agisce sul gradiente già calcolato, limitandone la norma globale; non modifica le Jacobiane locali e non recupera un gradiente svanito.  
**Correzione concreta:** attribuire a normalizzazione/residui il controllo della propagazione; descrivere il clipping solo come limite post-backward sugli aggiornamenti esplosivi.

### Il lab M02 reintroduce il falso sintomo di `zero_grad()`
**File:** `labs/lab_02_2_autograd.py` · **Sezione:** 5, accumulo dei gradienti  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il lab stampa «la loss SALE invece di scendere». La stessa pagina M02 e M03 spiegano correttamente che non esiste quel sintomo universale: la loss può oscillare, saturare o divergere. Il lab contraddice la correzione nel testo.  
**Correzione concreta:** sostituire la frase con l'effetto garantito — i gradienti includono la storia precedente e l'aggiornamento non è quello previsto — e rimandare ai possibili sintomi non univoci mostrati in M03.

### L'equivalenza dell'accumulo è presentata senza le condizioni necessarie
**File:** `m08-optimizer.html` · **Sezione:** risposta iniziale e batch effettivo  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** «identico cifra per cifra» vale nel giocattolo, con loss additiva, micro-batch della stessa dimensione e nessuna operazione dipendente dal batch. Non è universale con normalizzazioni o stato dipendenti dal batch, denominatori diversi per numero di token validi, dropout/ordine RNG e aritmetica in precisione finita. Anche «omettere la divisione equivale a moltiplicare il learning rate» è esatto solo per l'update SGD lineare; con AdamW il riscalamento può essere in gran parte normalizzato dai momenti e interagisce con epsilon, clipping e weight decay.  
**Correzione concreta:** limitare entrambe le equivalenze alle condizioni del lab. Per sequenze con maschere variabili, normalizzare per il numero complessivo di token supervisionati; per AdamW descrivere l'omissione come riscalamento del gradiente, non del learning rate in generale.

### Gli output HTML non corrispondono ai lab eseguiti
**File:** `m02-gradienti.html`, `m03-addestrare.html`, `m08-optimizer.html`, `m09-transformer.html`, `m13-cpt-sft.html`, `m14-lora.html`, `m15-finetuning.html` · **Sezione:** blocchi di output numerico  
**Severità:** BLOCKER  
**Cosa è sbagliato:** il vincolo centrale HTML↔lab è violato in sette moduli. Con l'ambiente dichiarato:

- M02 scrive 23 discese su 40; il lab stampa 22/40.
- M03 riporta a passo 20.000 `0.001747/0.057521`, 298× e 2,6×; il lab stampa `0.001765/0.056741`, 295× e 2,5×. Anche le righe intermedie differiscono.
- M08 riporta per warmup, fra gli altri, `112641.71/260.01` e `191584160/3003.94`; il lab stampa `112643.40/332.79` e `191583200/16551.92`.
- M09 riporta scarti attention/GQA `2.384e-07`; il lab stampa `1.192e-07`. Differiscono anche due valori RoPE e la media LayerNorm (`-1.24e-09` contro `-2.02e-09`). La pagina dice di avere confrontato `F.rms_norm`, ma su 2.2.2 il lab usa il fallback; il ramo nativo è non verificabile.
- M13 riporta SFT `321/96`, rapporto 4,6×, PPL `87506.30` e `676.06`; il lab stampa `214/64`, 6,9×, `87560.12` e `694.15`.
- M14 riporta merge `8.882e-16` e DoRA `1.110e-16`; il lab stampa `2.220e-15` e `2.776e-17`.
- M15 fissa tre scarti `0.18/0.21/0.19`, ma quella sezione usa ordine casuale non seminato e cambia a ogni esecuzione; in questa esecuzione stampa `0.1084/0.5835/0.1389`. Anche hash e dimensioni sono diversi: pagina `22e1f8a0…/9c4b7d31…`, 58/59 byte; lab `1c8896a9…/71db995b…`, 60/61 byte.

Parte degli ultimi bit dipende da piattaforma, backend e versione: proprio per questo non può essere presentata come output stabile senza registrare ambiente e tolleranza.  
**Correzione concreta:** rigenerare tutti i blocchi dall'output corrente nello stack dichiarato. Rendere deterministiche le sezioni destinate alla pagina; per scarti numerici dipendenti dal backend, riportare una tolleranza/ordine di grandezza e l'ambiente invece di un valore universale. Impedire la pubblicazione quando output catturato e HTML divergono.

### La notazione scientifica del lab M01 cambia il numero mostrato
**File:** `labs/lab_01_1_forme.py` · **Sezione:** 3, byte  
**Severità:** NITPICK  
**Cosa è sbagliato:** il lab stampa `3e+10 × 4 byte = 108.0 GB`: `3e+10 × 4` vale 120 GB, non 108. La variabile è correttamente 27 miliardi; è il formato `.0e` che la arrotonda a una cifra significativa.  
**Correzione concreta:** stampare `27e9`, `2.7e10` o più cifre significative, così fattori e risultato restano coerenti.

### Continuare i position ID non cambia la RoPE pura dentro un documento isolato
**File:** `m11-dataset.html` · **Sezione:** packing, maschera per documento e `position_ids`  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** con RoPE standard il prodotto `⟨R_m q, R_n k⟩` dipende da `m−n`. Se il secondo documento è isolato da una maschera e tutti i suoi position ID ricevono lo stesso offset, le differenze interne non cambiano; quindi i punteggi di attention non cambiano, salvo arrotondamento. M11 afferma invece che continuare la numerazione rende le posizioni relative sbagliate e cambia direttamente il calcolo, contraddicendo M09. Implementazioni con scaling o altri meccanismi posizionali possono comportarsi diversamente.  
**Correzione concreta:** per RoPE pura distinguere l'offset comune, invariabile, dalle distanze interne. Mantenere il controllo sullo stack reale, qualificandolo come dipendente dall'implementazione e da eventuali trasformazioni/scaling dei position ID.

### M13 trasforma un esito del giocattolo in una separazione universale CPT/SFT
**File:** `m13-cpt-sft.html`, `labs/lab_13_1_cpt_sft.py` · **Sezione:** tabella CPT/SFT e «Come si sceglie»  
**Severità:** BLOCKER  
**Cosa è sbagliato:** «SFT non insegna conoscenza nuova» e «nessun learning rate lo risolve» non seguono dal lab e sono falsi come regola generale. La stessa esecuzione mostra 100% sulle otto risposte fattuali presenti nell'SFT: il segnale supervisionato può insegnare fatti. Lo 0% riguarda soltanto quattro fatti mai mostrati nel formato Q/R a questo piccolo modello. Inoltre la PPL sulle stringhe grezze non misura da sola se un fatto è conosciuto o accessibile con altre formulazioni.  
**Correzione concreta:** limitare la conclusione al regime misurato: in questo lab CPT non trasferisce automaticamente quattro fatti a una formulazione Q/R mai vista, e SFT offre meno token supervisionati su questo dataset. Presentare CPT vs SFT come scelta empirica di obiettivo/formato, non come impossibilità di insegnare conoscenza via SFT.

### Il conto QLoRA è corretto per il quantizzatore giocattolo ma incoerente col totale M06
**File:** `m06-memoria.html`, `m14-lora.html`, `labs/lab_14_1_lora.py` · **Sezione:** QLoRA e memoria statica  
**Severità:** BLOCKER  
**Cosa è sbagliato:** `0,53125 = 0,5 + 2/64` byte/parametro e `27e9 × 2/64 = 843,75 MB` sono aritmeticamente corretti per l'ipotesi dichiarata: 4 bit più una scala fp16 ogni 64 valori. Il lab usa però un quantizzatore uniforme simmetrico semplificato, non misura un'implementazione QLoRA reale; metadati, codici, doppia quantizzazione e dtype delle scale dipendono da libreria e versione. Quindi «costo reale» e degrado 10,8% non sono trasferibili. Inoltre M14 dice di usare 0,531 in M06 ma continua a citare 15,4 GB, calcolati con 0,5: applicando 0,53125 alle stesse altre voci il totale è circa 16,23 GB.  
**Correzione concreta:** chiamare 0,531 e 840 MB «conto del quantizzatore semplificato». Per lo stack QLoRA scelto, contare i buffer effettivi e dichiarare versione/configurazione; aggiornare M06 e M14 con la stessa ipotesi, ottenendo 16,23 GB se restano invariati tutti gli altri termini.

### Il merge LoRA è equivalente in algebra, non identico in precisione finita
**File:** `m14-lora.html`, `labs/lab_14_1_lora.py` · **Sezione:** «Il merge, dimostrato»  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il lab misura uno scarto non nullo (`2.220e-15` in questa esecuzione) ma pagina e lab lo chiamano «stessa identica funzione» e «zero macchina». L'identità vale in aritmetica esatta; l'ordine diverso delle moltiplicazioni produce arrotondamento. M20 mostra poi che dtype e ordine del merge possono amplificare lo scarto.  
**Correzione concreta:** dire «algebricamente equivalente» e verificare lo scarto entro una tolleranza dichiarata per dtype/backend. Riservare «identico bit per bit» soltanto a un confronto che dia davvero uguaglianza esatta.

### I semi non garantiscono da soli identità bit per bit
**File:** `m15-finetuning.html`, `labs/lab_15_1_riproducibile.py` · **Sezione:** risposta iniziale  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** «sì, esattamente zero se hai fissato tutti i semi» generalizza un risultato del lab CPU. Su GPU possono restare kernel non deterministici; anche con operazioni deterministiche contano versione, hardware, backend e configurazione. La pagina lo riconosce più avanti, ma la risposta iniziale resta categorica e contraddittoria.  
**Correzione concreta:** attribuire l'identità bit per bit soltanto al lab CPU e all'ambiente fissato. Per il caso generale distinguere semi controllati, algoritmi deterministici e equivalenza statistica; dichiarare dipendenze di versione/backend.

### La spiegazione del calo di `logp(preferita)` in DPO omette il gradiente diretto
**File:** `m16-dpo.html`, `labs/lab_16_1_dpo.py` · **Sezione:** formula DPO e risultato  
**Severità:** BLOCKER  
**Cosa è sbagliato:** per `z=β[(logπ_w−logπref_w)−(logπ_l−logπref_l)]`, si ha `∂L/∂logπ_w = −βσ(−z) < 0` e `∂L/∂logπ_l = +βσ(−z) > 0`. Considerate come variabili indipendenti, la discesa del gradiente spinge quindi la preferita verso l'alto e la rifiutata verso il basso. È vero che l'obiettivo non garantisce un aumento assoluto: log-probabilità e parametri sono accoppiati dal softmax e condivisi fra token/sequenze, quindi un update parametrico può abbassarle entrambe. «Non c'è nulla che spinga verso l'alto» è però la spiegazione sbagliata del fenomeno osservato.  
**Correzione concreta:** mostrare le due derivate parziali; poi spiegare che l'aggiornamento avviene nello spazio dei parametri condivisi e delle probabilità normalizzate, non su due log-probabilità indipendenti. Distinguere «segnale locale verso l'alto» da «nessuna garanzia sull'esito assoluto dopo l'update».

### Il lab DPO non dimostra che il modello generi peggio o «meno qualunque cosa»
**File:** `m16-dpo.html`, `labs/lab_16_1_dpo.py` · **Sezione:** risultato che sorprende  
**Severità:** BLOCKER  
**Cosa è sbagliato:** il lab misura soltanto la log-probabilità totale di quattro stringhe preferite e quattro rifiutate. Non genera output, non misura qualità held-out, distribuzione delle lunghezze, EOS o massa spostata verso altre risposte. Una distribuzione normalizzata non può diventare meno propensa a produrre «qualunque cosa»: la massa va altrove. Il calo delle quattro stringhe non prova «genera peggio».  
**Correzione concreta:** limitare la conclusione a «le quattro risposte preferite esatte diventano meno probabili nonostante il margine salga». Per parlare di qualità o degenerazione, aggiungere una misura/generazione indipendente; altrimenti scrivere esplicitamente «non verificabile».

### DPO richiede un riferimento matematico, non sempre due modelli residenti
**File:** `m16-dpo.html` · **Sezione:** formula e memoria  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** «in memoria ce ne stanno due» è una scelta implementativa. Su un dataset fisso le log-probabilità del riferimento possono essere precalcolate; con adattatori si può condividere la base e cambiare stato dell'adattatore. Il costo dipende dalla libreria/versione. Anche «nessuna quantità di coppie insegna un fatto ignoto» è universale: il testo delle risposte preferite contiene segnale fattuale, sebbene meno diretto dell'SFT.  
**Correzione concreta:** dire che DPO richiede i log-ratio rispetto a una policy di riferimento; specificare che il lab usa due copie, mentre memoria e forward del riferimento dipendono dall'implementazione. Sostituire l'impossibilità sulla conoscenza con un confronto di efficienza/segnale da verificare sul compito.

### L'esperimento MoE non identifica causalmente il costo dello sbilanciamento
**File:** `m19-moe.html`, `labs/lab_19_1_moe.py` · **Sezione:** bilanciamento  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il router libero non collassa. Per ottenere la terza riga il lab cambia il problema imponendo `forza=8.0` ai logit del router e riaddestra. La differenza di MSE fra run libero e forzato non isola lo sbilanciamento come causa; confronta traiettorie/configurazioni diverse, su un solo seed. Analogamente un solo run non prova che il termine di bilanciamento peggiori di 6× un MoE già bilanciato in generale.  
**Correzione concreta:** presentare le quattro righe come scenari costruiti che mostrano l'aspetto delle metriche, non come stime causali di qualità. Per una conclusione causale servono condizioni abbinate, più seed e intervalli secondo M12; senza questi dati, scrivere «non verificabile».

### La percentuale dell'esperto top conta assegnazioni, non token
**File:** `m19-moe.html`, `labs/lab_19_1_moe.py` · **Sezione:** metriche del routing  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** con routing top-2 il lab conta due assegnazioni per token e normalizza per `c.sum() = 2N`. Un valore 50% significa 50% delle assegnazioni; può corrispondere allo stesso esperto selezionato per il 100% dei token. Pagina e lab lo chiamano invece «50% dei token» e «frazione di token per esperto».  
**Correzione concreta:** rinominare la misura «quota delle assegnazioni top-k»; oppure dividere per N e riportare la frazione di token che seleziona ciascun esperto, chiarendo che le quote sommano a k.

### Il fattore 3.000× di M20 non dimostra accumulo o degrado multistrato
**File:** `m20-parity.html`, `labs/lab_20_1_parity.py` · **Sezione:** merge in precisione ridotta  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** sul singolo strato, seed e scala del lab il rapporto è circa 3.241× (`7.727e-3 / 2.384e-6`), quindi «3.000×» è un arrotondamento fedele della misura richiesta. Non segue però che su decine di strati l'errore si accumuli monotonicamente o peggiori la qualità: può propagarsi, cancellarsi o amplificarsi a seconda di pesi, attivazioni e backend. Il lab non misura una rete multistrato né qualità.  
**Correzione concreta:** limitare il fattore al singolo strato/configurazione misurati. Presentare l'effetto end-to-end come non verificabile da questo lab, mantenendo il controllo di parity sull'intero modello come test operativo.

### «IC contiene zero» non significa assenza del fenomeno
**File:** `m21-diagnosi.html` · **Sezione:** passo zero e risposte finali  
**Severità:** BLOCKER  
**Cosa è sbagliato:** M21 dice «non hai un fenomeno da spiegare». M12 dice correttamente il contrario: un intervallo che contiene zero è compatibile sia con assenza di effetto sia con effetto reale non rilevato. La simulazione del corso ha una differenza vera di 3,3 punti e con 500 esempi la rileva solo nel 32% dei casi. Confondere mancato rigetto con prova di assenza produce diagnosi false.  
**Correzione concreta:** sostituire la conclusione con «non hai evidenza sufficiente per attribuire una causa». Decidere prima la differenza minima rilevante e la precisione richiesta; se l'intervallo è troppo largo, aumentare o ridisegnare la valutazione. Fermarsi soltanto rispetto all'attribuzione causale, non dichiarare inesistente l'effetto.

### Il bootstrap percentile è nominale, non garantito «per costruzione»
**File:** `m12-eval.html`, `labs/lab_12_1_eval.py` · **Sezione:** bootstrap e regola del tre  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** la regola del tre è corretta sotto il modello di 200 prove Bernoulli indipendenti: con 0 eventi, `3/200 = 1,5%` approssima l'estremo superiore 95% (quello esatto è circa 1,49%). Non è invece corretto trasferire questo risultato alla frase generale che il tasso di conclusioni invertite del bootstrap percentile sia «limitato per costruzione». Quello del lab è un intervallo percentile discreto e la copertura effettiva dipende dalla distribuzione e dalla dimensione campionaria; 95% è nominale, non una garanzia universale.  
**Correzione concreta:** mantenere l'1,5% come limite sul tasso della specifica simulazione ripetuta, con assunzione di prove indipendenti e stesso regime. Chiamare l'intervallo «bootstrap percentile nominale 95%» e non dedurne copertura universale.

### Il confronto prima/dopo di M12 non usa gli stessi esempi
**File:** `labs/lab_12_1_eval.py`, `m12-eval.html` · **Sezione:** 7, catastrophic forgetting  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** il lab valuta «prima» con `seed=1` e «dopo» con `seed=2`, quindi cambia anche le difficoltà degli esempi. È contrario alla motivazione appaiata appena dimostrata e il delta combina cambiamento di abilità e rumore del campione.  
**Correzione concreta:** generare una sola lista di difficoltà/esempi per capacità e applicarla a entrambi i valori prima/dopo; calcolare anche l'intervallo appaiato della differenza.

### L'ordine per costo di M21 è una euristica non misurata
**File:** `m21-diagnosi.html` · **Sezione:** nove ipotesi ordinate  
**Severità:** SHOULD-FIX  
**Cosa è sbagliato:** minuti/ore/giorni/settimane e quindi l'ordine non derivano da un lab. Dipendono da artefatti disponibili, automazione, dimensione dei run, stack e organizzazione. Per esempio, una valutazione già pronta può costare meno di ricostruire una parity priva di manifest; un rank sweep piccolo può costare meno di un audit dati manuale. Presentare l'ordine come fisso trasferisce un giudizio del caso di studio ad altri regimi.  
**Correzione concreta:** dichiarare la lista come ordine iniziale del caso di studio, non universale. Far stimare al lettore costo e potere discriminante di ogni ipotesi nel proprio contesto, poi ordinare con quei valori. Dove mancano dati di costo, scrivere «non verificabile».
