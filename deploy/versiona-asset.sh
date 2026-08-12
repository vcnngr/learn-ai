#!/bin/sh
# Timbra la versione negli URL degli asset: assets/x.css -> assets/x.css?v=TAG
#
# Sta in uno script e non inline nel Dockerfile perche' l'espressione
# deve contenere il carattere " e attraversare due livelli di quoting
# (Docker RUN + shell): inline si rompeva in silenzio, e il build
# falliva sul controllo finale invece che sulla sed.
set -eu
VERSIONE="${1:?serve la versione}"
DIR="${2:-/usr/share/nginx/html}"

for f in "$DIR"/*.html; do
  # delimitatore # e non |: il | serve gia' all'alternanza (css|js),
  # e usarlo per entrambi confonde sed prima ancora di compilare la regex
  sed -i -E 's#(assets/[a-z]+\.(css|js))"#\1?v=SEGNAPOSTO"#g' "$f"
  sed -i "s#?v=SEGNAPOSTO#?v=${VERSIONE}#g" "$f"
done

# Se la marcatura non ha attecchito il build deve fermarsi qui: un
# rilascio con URL non versionati riporta il difetto che questo script
# esiste per chiudere, e lo farebbe in silenzio.
grep -q "style.css?v=${VERSIONE}" "$DIR/index.html"
grep -q "tema.js?v=${VERSIONE}"   "$DIR/index.html"
echo "asset versionati: v=${VERSIONE}"
