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

# --- terza iniezione: la regola giusta sul SELETTORE SBAGLIATO ---
#
# E' il difetto che una review ha dovuto trovare al posto del gate: la
# versione precedente cercava la stringa «da qualche parte nel media
# query», quindi bastava che `order: -1` esistesse su un selettore
# qualunque. Qui lo sposto su .sidebar-inner e il gate deve continuare
# ad accusare .sidebar-piede.
BK3=$(mktemp); cp "$CSSF" "$BK3"
python3 - "$CSSF" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text()
t = t.replace("    order: -1;\n", "", 1)
t = t.replace("  .sidebar-inner { flex: none; min-height: auto; overflow: visible; padding-top: 22px; }",
              "  .sidebar-inner { flex: none; min-height: auto; overflow: visible; padding-top: 22px; order: -1; }")
p.write_text(t)
PYEOF
OUT5=$(python3 deploy/prova-layout.py 2>&1) && ESITO5=0 || ESITO5=$?
cp "$BK3" "$CSSF"; rm -f "$BK3"
SELETTORE=0
if printf '%s' "$OUT5" | grep "il piede va in testa" | grep -q "MANCA su .sidebar-piede"; then SELETTORE=1; fi
printf "  %-44s %s\n" "layout: regola sul selettore sbagliato" "$([ $ESITO5 = 1 ] && [ $SELETTORE = 1 ] && echo "vista" || echo NO)"

# --- quarta: falso verde da SOTTOSTRINGA ---
# `flex-direction: column` e' contenuto in `column-reverse`, che
# ribalta la colonna e porta il piede in cima sul desktop: un layout
# diverso, che il confronto per sottostringa accettava.
BK4=$(mktemp); cp "$CSSF" "$BK4"
sed 's/flex-direction: column;/flex-direction: column-reverse;/' "$BK4" > "$CSSF"
OUT6=$(python3 deploy/prova-layout.py 2>&1) && ESITO6=0 || ESITO6=$?
cp "$BK4" "$CSSF"
SOTTOSTRINGA=0
if printf '%s' "$OUT6" | grep "colonna flex" | grep -q "column-reverse"; then SOTTOSTRINGA=1; fi
printf "  %-44s %s\n" "layout: valore contenuto ma diverso" "$([ $ESITO6 = 1 ] && [ $SOTTOSTRINGA = 1 ] && echo "vista" || echo NO)"

# --- quinta: falso verde da CASCATA ---
# In CSS vince l'ultima dichiarazione. Una regola successiva che
# sovrascrive `flex: none` passava, perche' il gate si fermava alla
# prima regola col selettore giusto.
printf '\n.sidebar-piede { flex: 1; }\n' >> "$CSSF"
OUT7=$(python3 deploy/prova-layout.py 2>&1) && ESITO7=0 || ESITO7=$?
cp "$BK4" "$CSSF"; rm -f "$BK4"
CASCATA=0
if printf '%s' "$OUT7" | grep "non si comprime" | grep -q "flex vale"; then CASCATA=1; fi
printf "  %-44s %s\n" "layout: sovrascritto da una regola dopo" "$([ $ESITO7 = 1 ] && [ $CASCATA = 1 ] && echo "vista" || echo NO)"

# --- sesta: override per SPECIFICITA' ---
# `.wrap .sidebar-piede` vince nel browser e il modello del gate non
# valuta la specificita'. Deve dichiararlo NON DETERMINABILE, non "ok".
BK5=$(mktemp); cp "$CSSF" "$BK5"
printf '\n.wrap .sidebar-piede { flex: 1; }\n' >> "$CSSF"
OUT8=$(python3 deploy/prova-layout.py 2>&1) && ESITO8=0 || ESITO8=$?
cp "$BK5" "$CSSF"
SPECIF=0
if printf '%s' "$OUT8" | grep "non si comprime" | grep -q "NON DETERMINABILE"; then SPECIF=1; fi
printf "  %-44s %s\n" "layout: override per specificita'" "$([ $ESITO8 = 1 ] && [ $SPECIF = 1 ] && echo "dichiarato" || echo NO)"

# --- settima: !important PRIMA, valore normale DOPO ---
# Il browser tiene il primo. Un modello "vince l'ultima" direbbe il
# contrario del browser, cioe' sbaglierebbe in silenzio.
python3 - "$CSSF" <<'PYEOF'
import pathlib, sys
p = pathlib.Path(sys.argv[1]); t = p.read_text()
p.write_text(t.replace(".sidebar-piede {\n  flex: none;",
                       ".sidebar-piede {\n  flex: 1 !important;\n  flex: none;", 1))
PYEOF
OUT9=$(python3 deploy/prova-layout.py 2>&1) && ESITO9=0 || ESITO9=$?
cp "$BK5" "$CSSF"; rm -f "$BK5"
IMPORTANTE=0
if printf '%s' "$OUT9" | grep "non si comprime" | grep -q "flex vale '1'"; then IMPORTANTE=1; fi
printf "  %-44s %s\n" "layout: !important prima di una normale" "$([ $ESITO9 = 1 ] && [ $IMPORTANTE = 1 ] && echo "vista" || echo NO)"

# --- ottava: le forme valide di !important, in uno script a se' ---
# Sta separato perche' inietta cinque varianti e le verifica una per
# una: infilarlo qui dentro voleva dire annidare heredoc, che e' come
# ci si sbaglia.
python3 deploy/prova-important.py >/dev/null 2>&1 && VARIANTI=1 || VARIANTI=0
printf "  %-44s %s\n" "layout: ogni forma valida di !important" "$([ $VARIANTI = 1 ] && echo si || echo NO)"

OUT4=$(python3 deploy/prova-layout.py 2>&1) && ESITO4=0 || ESITO4=$?
TACE4=1
case "$OUT4" in *MANCA*) TACE4=0 ;; esac

printf "  %-44s %s\n" "layout: uscita col difetto (atteso 1)" "$ESITO3"
printf "  %-44s %s\n" "layout: nomina proprio order: -1" "$([ $NOMINA3 = 1 ] && echo si || echo NO)"
printf "  %-44s %s\n" "layout: non accusa gli altri invarianti" "$([ $ALTRI3 = 0 ] && echo si || echo NO)"
printf "  %-44s %s\n" "layout: a CSS integro tace" "$([ $TACE4 = 1 ] && [ $ESITO4 = 0 ] && echo si || echo NO)"

echo ""
if [ "$ESITO" = "1" ] && [ "$NOMINA" = "1" ] && [ "$ALTRE" = "0" ] && [ "$ESITO2" = "0" ] && [ "$TACE" = "1" ] \
   && [ "$ESITO3" = "1" ] && [ "$NOMINA3" = "1" ] && [ "$ALTRI3" = "0" ] && [ "$ESITO4" = "0" ] && [ "$TACE4" = "1" ] \
   && [ "$ESITO5" = "1" ] && [ "$SELETTORE" = "1" ] \
   && [ "$ESITO6" = "1" ] && [ "$SOTTOSTRINGA" = "1" ] \
   && [ "$ESITO7" = "1" ] && [ "$CASCATA" = "1" ] \
   && [ "$ESITO8" = "1" ] && [ "$SPECIF" = "1" ] \
   && [ "$ESITO9" = "1" ] && [ "$IMPORTANTE" = "1" ] && [ "$VARIANTI" = "1" ]; then
  echo "Il gate vede il difetto e tace quando non c'e'."
  exit 0
fi
echo "IL GATE NON DIMOSTRA NIENTE: il verde che produce non significa nulla."
exit 1
