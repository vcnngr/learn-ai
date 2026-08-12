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

echo ""
if [ "$ESITO" = "1" ] && [ "$NOMINA" = "1" ] && [ "$ALTRE" = "0" ] && [ "$ESITO2" = "0" ] && [ "$TACE" = "1" ]; then
  echo "Il gate vede il difetto e tace quando non c'e'."
  exit 0
fi
echo "IL GATE NON DIMOSTRA NIENTE: il verde che produce non significa nulla."
exit 1
