#!/bin/sh
# Ogni tipo di risposta porta le intestazioni che deve portare?
#
#   sh deploy/prova-intestazioni.sh vcnngr/learn-ai-sito:TAG
#
# Nasce da quattro difetti consecutivi, tutti la stessa trappola: in
# nginx un add_header dentro un location AZZERA quelli ereditati dal
# server. Un elenco ripetuto a mano in piu' punti produce sviste; un
# controllo che le guarda tutte le trova prima del revisore.
#
# Esce 1 se una risposta manca di una intestazione o ha la politica di
# cache sbagliata.
set -eu
IMG="$1"
# porta libera scelta al volo: una prova precedente rimasta appesa
# faceva fallire questa con "port is already allocated", che non ha
# niente a che vedere con le intestazioni
PORTA=$(python3 -c "import socket;s=socket.socket();s.bind(('',0));print(s.getsockname()[1]);s.close()")
NOME="prova-intestazioni-$$"

docker run -d --rm --name "$NOME" -p "$PORTA:8080" "$IMG" >/dev/null
trap 'docker stop "$NOME" >/dev/null 2>&1 || true' EXIT
sleep 2

B="http://localhost:$PORTA"
CSS=$(curl -s "$B/" | grep -o 'assets/style\.[a-z0-9]*\.css' | head -1)
if [ -z "$CSS" ]; then
  echo "ERRORE: nessun asset col digest nel nome" >&2
  exit 1
fi

falliti=0
printf "%-30s %-4s %-4s %-4s %s\n" "risposta" "CSP" "nosn" "refP" "cache"

controlla() {
  u="$1"; atteso="$2"
  H=$(curl -sI "$B$u")
  c=$(printf '%s' "$H" | grep -ic content-security-policy || true)
  n=$(printf '%s' "$H" | grep -ic x-content-type-options || true)
  r=$(printf '%s' "$H" | grep -ic referrer-policy || true)
  cc=$(printf '%s' "$H" | grep -i '^cache-control' | tr -d '\r' | cut -d' ' -f2- || true)
  nota=""
  if [ "$c$n$r" != "111" ]; then nota="  <- INTESTAZIONE MANCANTE"; fi
  case "$cc" in
    *"$atteso"*) ;;
    *) nota="$nota  <- CACHE: atteso $atteso" ;;
  esac
  printf "%-30s %-4s %-4s %-4s %s%s\n" "$u" "$c" "$n" "$r" "$cc" "$nota"
  if [ -n "$nota" ]; then falliti=$((falliti + 1)); fi
}

controlla "/"                            "max-age=300"
controlla "/$CSS"                        "immutable"
controlla "/assets/style.deadbeef.css"   "no-store"
controlla "/nonesiste.html"              "no-store"

echo ""
if [ "$falliti" -gt 0 ]; then
  echo "$falliti risposte non conformi."
  echo "Ricorda: add_header in un location AZZERA quelli del server."
  exit 1
fi
echo "Tutte le risposte portano CSP, nosniff e Referrer-Policy, con la"
echo "politica di cache attesa. Gli errori non sono cacheabili."
