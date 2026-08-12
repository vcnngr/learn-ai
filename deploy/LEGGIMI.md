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

## Cache: perché gli asset portano il digest nel nome

`assets/style.css` non esiste più. Al build ogni asset viene rinominato col
digest del proprio contenuto — `style.c109b08f.css` — e i riferimenti nelle
pagine riscritti. Il sorgente in git resta pulito: la marcatura vive solo dentro
l'immagine.

**Prima soluzione, insufficiente.** Avevo usato una query `?v=<tag>`. Non regge
il rollout: per qualche secondo convivono pod vecchi e nuovi, e una pagina
servita dal pod nuovo che chiede `style.css?v=NUOVO` a un pod vecchio riceve il
file vecchio — nginx la query la ignora. Stesso difetto, spostato dentro la
finestra di aggiornamento.

**Perché il digest regge.** Contenuto diverso ⇒ nome diverso ⇒ URL diverso. Un
pod vecchio quel file non ce l'ha e risponde **404**: rumoroso, e si risolve da
solo al ricarico. Un CSS stantio invece è silenzioso, e davanti c'è Cloudflare
che riscrive `Cache-Control` e se lo tiene per ore.

Con nomi immutabili gli asset possono avere `max-age=31536000, immutable`: un URL
identifica un contenuto e basta.

**Cosa NON è risolto.** La finestra di rollout non è zero. `maxUnavailable: 0`
la riduce al tempo che un pod vecchio impiega a uscire dal Service, ma in quei
secondi una richiesta può ancora prendere 404 su un asset. È un compromesso
scelto: 404 che si autocorregge invece di contenuto sbagliato che persiste.

**Residuo noto.** Dopo il passaggio al digest, gli URL vecchi (`assets/style.css`)
restano in cache su Cloudflare finché non scadono — `cf-cache-status: HIT`, 404
all'origine. È innocuo perché nessuna pagina li referenzia più, e si estingue da
sé; ma se serve immediatezza, va purgata la cache CDN.

## Rilasciare una versione nuova## Rilasciare una versione nuova

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
