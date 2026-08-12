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

NOMINA=0
case "$OUT" in *"INTESTAZIONE MANCANTE"*) NOMINA=1 ;; esac

printf "  %-44s %s\n" "uscita col difetto (atteso 1)" "$ESITO"
printf "  %-44s %s\n" "il rapporto nomina il difetto" "$([ $NOMINA = 1 ] && echo si || echo NO)"

ripristina
trap - EXIT

# --- e a immagine integra deve tacere ---
docker build -q -f Dockerfile.sito -t learn-ai-sito:integra . >/dev/null
OUT2=$(sh deploy/prova-intestazioni.sh learn-ai-sito:integra 2>&1) && ESITO2=0 || ESITO2=$?
TACE=1
case "$OUT2" in *"INTESTAZIONE MANCANTE"*) TACE=0 ;; esac
docker rmi -f learn-ai-sito:integra >/dev/null 2>&1 || true

printf "  %-44s %s\n" "uscita a immagine integra (atteso 0)" "$ESITO2"
printf "  %-44s %s\n" "a immagine integra tace" "$([ $TACE = 1 ] && echo si || echo NO)"

echo ""
if [ "$ESITO" = "1" ] && [ "$NOMINA" = "1" ] && [ "$ESITO2" = "0" ] && [ "$TACE" = "1" ]; then
  echo "Il gate vede il difetto e tace quando non c'e'."
  exit 0
fi
echo "IL GATE NON DIMOSTRA NIENTE: il verde che produce non significa nulla."
exit 1
