#!/bin/sh
# La prova delle intestazioni sa fallire?
#
#   sh deploy/prova-la-prova.sh
#
# Stessa regola del passo 0 di corso/RILASCIO.md, applicata al gate del
# sito: un controllo che non si e' mai visto fallire non e' un
# controllo. `prova-intestazioni.sh` era stato verificato UNA VOLTA a
# mano, e la verifica buttata via — cioe' il difetto che prova_i_gate.py
# esiste per impedire, ripetuto nel gate aggiunto subito dopo.
#
# Come prova_i_gate.py, non guarda solo il codice di uscita: verifica
# che il rapporto NOMINI il difetto iniettato. Un gate gia' rosso per
# altri motivi darebbe 1 comunque, e la prova passerebbe a vuoto.
set -eu
QUI=$(cd "$(dirname "$0")/.." && pwd)
cd "$QUI"

CONF="deploy/intestazioni-sicurezza.conf"
BACKUP=$(mktemp)
cp "$CONF" "$BACKUP"
ripristina() { cp "$BACKUP" "$CONF"; rm -f "$BACKUP"; docker rmi -f learn-ai-sito:rotta >/dev/null 2>&1 || true; }
trap ripristina EXIT

echo "Provo che il gate delle intestazioni sappia fallire."
echo ""

# --- difetto iniettato: tolgo Referrer-Policy, che e' esattamente cio'
#     che una review ha dovuto trovare al posto mio ---
grep -v "Referrer-Policy" "$BACKUP" > "$CONF"
docker build -q -f Dockerfile.sito -t learn-ai-sito:rotta . >/dev/null

OUT=$(sh deploy/prova-intestazioni.sh learn-ai-sito:rotta 2>&1) && ESITO=0 || ESITO=$?

# QUALE intestazione manca, non "ne manca una".
#
# Cercare la stringa generica INTESTAZIONE MANCANTE non dimostrava
# niente: se il gate avesse segnalato un CSP assente al posto del
# Referrer-Policy che ho tolto, la prova sarebbe passata lo stesso. E'
# la stessa debolezza per cui prova_i_gate.py era gia' stato corretto —
# "il controllo era rosso" contro "il controllo ha visto QUESTO".
#
# Il rapporto stampa tre colonne, CSP / nosniff / Referrer-Policy. La
# firma del difetto iniettato e' quindi "1 1 0": le prime due presenti,
# la terza no. Un difetto diverso produce una firma diversa.
FIRMA='1  *1  *0 '
NOMINA=0
if printf '%s' "$OUT" | grep -qE "$FIRMA"; then NOMINA=1; fi

# e non deve segnalare le ALTRE due come mancanti
ALTRE=0
if printf '%s' "$OUT" | grep -qE '(0  *[01]  *[01] |[01]  *0  *[01] )'; then ALTRE=1; fi

printf "  %-44s %s\n" "uscita col difetto (atteso 1)" "$ESITO"
printf "  %-44s %s\n" "firma 1/1/0: manca proprio Referrer-Policy" "$([ $NOMINA = 1 ] && echo si || echo NO)"
printf "  %-44s %s\n" "non accusa CSP o nosniff" "$([ $ALTRE = 0 ] && echo si || echo NO)"

ripristina
trap - EXIT

# --- e a immagine integra deve tacere ---
docker build -q -f Dockerfile.sito -t learn-ai-sito:integra . >/dev/null
OUT2=$(sh deploy/prova-intestazioni.sh learn-ai-sito:integra 2>&1) && ESITO2=0 || ESITO2=$?
TACE=1
if printf '%s' "$OUT2" | grep -qE "$FIRMA"; then TACE=0; fi
case "$OUT2" in *"INTESTAZIONE MANCANTE"*) TACE=0 ;; esac
docker rmi -f learn-ai-sito:integra >/dev/null 2>&1 || true

printf "  %-44s %s\n" "uscita a immagine integra (atteso 0)" "$ESITO2"
printf "  %-44s %s\n" "a immagine integra tace" "$([ $TACE = 1 ] && echo si || echo NO)"

# ----------------------------------------------------------------
# E il gate degli invarianti di impaginazione?
#
# Stessa regola, stessa forma: iniettare il difetto e verificare che il
# rapporto NOMINI quell'invariante, non che diventi genericamente rosso.
echo ""
CSSF="corso/assets/style.css"
BK2=$(mktemp); cp "$CSSF" "$BK2"
# tolgo l'order: -1 del breakpoint, cioe' il difetto vero che una review
# ha dovuto trovare: il comando che sotto i 980px torna in fondo
sed 's/    order: -1;//' "$BK2" > "$CSSF"
OUT3=$(python3 deploy/prova-layout.py 2>&1) && ESITO3=0 || ESITO3=$?
cp "$BK2" "$CSSF"; rm -f "$BK2"

# Il confronto va fatto sulla STESSA RIGA. Un glob come
# *"colonna flex"*"MANCA"* combacia anche quando le due stringhe stanno
# su righe diverse — «colonna flex ok» sopra e «MANCA» piu' sotto — e
# accusava un invariante sano. E' la stessa imprecisione della firma
# generica: cercare due cose vicine invece della cosa giusta.
NOMINA3=0
if printf '%s' "$OUT3" | grep "il piede va in testa" | grep -q "MANCA"; then NOMINA3=1; fi
ALTRI3=0
if printf '%s' "$OUT3" | grep -E "colonna flex|scorre l.indice|non si comprime" | grep -q "MANCA"; then ALTRI3=1; fi

OUT4=$(python3 deploy/prova-layout.py 2>&1) && ESITO4=0 || ESITO4=$?
TACE4=1
case "$OUT4" in *MANCA*) TACE4=0 ;; esac

printf "  %-44s %s\n" "layout: uscita col difetto (atteso 1)" "$ESITO3"
printf "  %-44s %s\n" "layout: nomina proprio order: -1" "$([ $NOMINA3 = 1 ] && echo si || echo NO)"
printf "  %-44s %s\n" "layout: non accusa gli altri invarianti" "$([ $ALTRI3 = 0 ] && echo si || echo NO)"
printf "  %-44s %s\n" "layout: a CSS integro tace" "$([ $TACE4 = 1 ] && [ $ESITO4 = 0 ] && echo si || echo NO)"

echo ""
if [ "$ESITO" = "1" ] && [ "$NOMINA" = "1" ] && [ "$ALTRE" = "0" ] && [ "$ESITO2" = "0" ] && [ "$TACE" = "1" ] \
   && [ "$ESITO3" = "1" ] && [ "$NOMINA3" = "1" ] && [ "$ALTRI3" = "0" ] && [ "$ESITO4" = "0" ] && [ "$TACE4" = "1" ]; then
  echo "Il gate vede il difetto e tace quando non c'e'."
  exit 0
fi
echo "IL GATE NON DIMOSTRA NIENTE: il verde che produce non significa nulla."
exit 1
