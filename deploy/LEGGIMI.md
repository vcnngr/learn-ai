# Il sito sul cluster

`https://corso.vcnngr.it` — namespace `default`, due repliche, nessun PVC.

## Perché nessun volume

Il contenuto sta **dentro l'immagine** ed è immutabile per build. Aggiornare il
sito significa costruire un tag nuovo, non scrivere su un volume: così quello che
è online corrisponde sempre a un commit. È la stessa disciplina del manifest di
M15, applicata al deploy.

## Due immagini, da non confondere

| File | Cosa contiene | A cosa serve |
|---|---|---|
| `Dockerfile` | python, torch, numpy | **ambiente di riferimento dei lab**: genera i numeri e fa girare il gate |
| `Dockerfile.sito` | nginx + HTML/CSS/JS | **quello che gira sul cluster**: serve le pagine, non esegue niente |

Sul cluster **non c'è Python**. I lab non sono nemmeno copiati nell'immagine: si
prendono da git, dove è giusto che stiano. Chiederli via web dà 404, di proposito.

## Cache: perché gli URL sono versionati

Davanti al cluster c'è **Cloudflare**, e riscrive le intestazioni di cache
dell'origine: un `Cache-Control: no-cache` diventa `max-age=14400` al bordo. Con
URL stabili una release può quindi accoppiare **HTML nuovo e CSS vecchio**.

Non è un rischio teorico: è successo al primo rilascio del toggle. Le pagine sono
uscite senza separatori perché il CSS in cache non aveva ancora le classi nuove,
e `cf-cache-status` diceva `HIT`.

`deploy/versiona-asset.sh` timbra la versione negli URL al momento del build —
`assets/style.css?v=<tag>` — e un URL diverso è una risorsa diversa per qualunque
cache, senza dover chiedere il permesso a nessuno. Il sorgente in git resta pulito:
la marcatura esiste solo dentro l'immagine. Lo script **fallisce il build** se la
marcatura non attecchisce, invece di produrre un'immagine con URL non versionati.

## Rilasciare una versione nuova

```bash
TAG="v$(date +%Y%m%d-%H%M)"
docker build --platform linux/amd64 -f Dockerfile.sito \
       --build-arg VERSIONE=$TAG -t vcnngr/learn-ai-sito:$TAG .
docker push vcnngr/learn-ai-sito:$TAG
sed -i '' "s|learn-ai-sito:.*|learn-ai-sito:$TAG|" deploy/k8s.yaml
kubectl apply -f deploy/k8s.yaml
kubectl -n default rollout status deploy/learn-ai
```

Prima però va passata la checklist di `corso/RILASCIO.md`: il sito pubblica i
numeri, e pubblicarli senza aver verificato che vengano dai lab vanifica il
vincolo su cui il corso è costruito.

## Com'è messo insieme

- **TLS**: il browser vede il certificato di **Cloudflare**, che fa da proxy
  (`188.114.96.7/97.7`). Il certificato di cert-manager (`learn-ai-tls`,
  ClusterIssuer `le-global-issuer`, DNS-01 su Cloudflare) copre il tratto fra
  Cloudflare e traefik. Sono due cose diverse e vanno tenute distinte.
- **Redirect**: due Ingress, non uno. Traefik lega un router a un entrypoint,
  quindi `websecure` e `web` sono due oggetti separati; il secondo porta il
  middleware `default-redirect-https@kubernetescrd`. Con un Ingress solo,
  `http://` risponde 404 invece di rimandare — ed è esattamente com'era al primo
  deploy.
- **Sicurezza**: rootfs in sola lettura, non-root (uid 101), tutte le capability
  eliminate. Le due scritture che nginx richiede — pid e cache — sono `emptyDir`
  in memoria, non volumi.
- **Intestazioni**: CSP, `nosniff`, `no-referrer`. Attenzione alla trappola di
  nginx: un `add_header` dentro un `location` **azzera** quelli ereditati dal
  blocco `server`. Per questo il caching usa `expires` e non `add_header`.
