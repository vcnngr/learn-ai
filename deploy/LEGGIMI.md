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

## Rilasciare una versione nuova

```bash
TAG="v$(date +%Y%m%d-%H%M)"
docker build --platform linux/amd64 -f Dockerfile.sito -t vcnngr/learn-ai-sito:$TAG .
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
