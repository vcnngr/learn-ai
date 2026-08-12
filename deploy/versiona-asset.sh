#!/bin/sh
# Rinomina ogni asset col DIGEST DEL SUO CONTENUTO e riscrive i
# riferimenti nelle pagine:  assets/style.css -> assets/style.a1b2c3d4.css
#
# PERCHE' L'HASH E NON UNA QUERY ?v=
#
# La query string non basta durante un rollout. Per qualche secondo
# convivono pod vecchi e nuovi: una pagina servita dal pod NUOVO chiede
# assets/style.css?v=NUOVO, quella richiesta puo' finire su un pod
# VECCHIO, e nginx serve il file statico ignorando la query — quindi
# HTML nuovo con CSS vecchio, in silenzio. E' lo stesso difetto che il
# versionamento doveva chiudere, spostato dentro la finestra di rollout.
#
# Con il digest nel NOME, un pod vecchio quel file non ce l'ha e
# risponde 404. Non e' gratis — per un istante una pagina puo' restare
# senza stile — ma un 404 e' RUMOROSO e si risolve da solo al ricarico,
# mentre un CSS stantio e' silenzioso e Cloudflare se lo tiene per ore.
#
# Cosa NON risolve: la finestra non e' zero. Con maxUnavailable: 0 dura
# quanto ci mette il pod vecchio a uscire dal Service. E' dichiarato in
# deploy/LEGGIMI.md invece di essere spacciato per risolto.
set -eu
DIR="${1:-/usr/share/nginx/html}"
A="$DIR/assets"

for f in "$A"/*.css "$A"/*.js; do
  [ -f "$f" ] || continue
  base=$(basename "$f")
  nome=${base%.*}
  est=${base##*.}
  case "$nome" in *.*) continue ;; esac        # gia' marcato, non rimarcare
  h=$(sha256sum "$f" | cut -c1-8)
  nuovo="${nome}.${h}.${est}"
  mv "$f" "$A/$nuovo"
  for p in "$DIR"/*.html; do
    sed -i "s#assets/${base}\"#assets/${nuovo}\"#g" "$p"
  done
  echo "  ${base} -> ${nuovo}"
done

# Se un riferimento non marcato sopravvive, il build deve fermarsi: una
# pagina che chiede ancora assets/style.css riporta il difetto intero.
if grep -qE 'assets/[a-z]+\.(css|js)"' "$DIR"/*.html; then
  echo "ERRORE: riferimenti non marcati rimasti:" >&2
  grep -ohE 'assets/[a-z]+\.(css|js)"' "$DIR"/*.html | sort -u >&2
  exit 1
fi
echo "asset marcati col digest del contenuto"
