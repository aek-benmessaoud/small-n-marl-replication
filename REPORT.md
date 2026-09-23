# Rapport final — Project09

**Titre :** Localisation-aware coverage multi-caméras sur MATE avec MAPPO
et récompense intrinsèque Chao-U (transposition de Project08).

**Question de recherche :** la récompense intrinsèque *Chao-U* (richesse
angulaire des relevés de visées par cible) améliore-t-elle la couverture /
localisation des caméras, comparé au signal env seul ?

**Réponse (bilan complet, 5 intrinsèques × 3 régimes, 24 campagnes) :**
**l'intrinsèque n'améliore pas la couverture en régime 4v8** (5 familles
testées — dense, chao discret, chao fenêtré, localization objective, RND —
aucune ne franchit p<0.05 sur la récompense ni le mean_seen). L'orthogonalité
diversité/couverture est un **artefact du régime** : en 8v8 (8 caméras pour
8 cibles), le chao fenêtré produit un gain hétérogène mais réel (C_end
fenêtré +0.148, 7/12, p=0.092 ; seq_loc 10/12), **spécifique à la diversité
angulaire** (RND neutre dans le même régime). Le mécanisme est « diversité
angulaire + excédent de caméras » — ni le régime découplé seul, ni la marge
visuelle, ni une exploration générale ne suffisent.

---

## 1. Cadre expérimental

| Élément | Valeur |
|---|---|
| Algorithme | MAPPO (PPO centralisé) décentralisé par caméra |
| Environnement | MATE 0.1.0, `MultiAgentTracking-v0`, régime `MATE-4v2-9-v0` (4 caméras, 2 cibles) |
| Stack | Python 3.10 (`.venv`), numpy 1.26.4, gym 0.21.0, torch 2.13.0+cpu |
| Seeds | `env_seed = BASE_SEED + seed*1000` ; `policy_seed = env_seed*1000+1` |
| Protocole | bras `intrinsic` vs `no_intrinsic`, même seed env, mêmes hyperparamètres |
| Stats | Wilcoxon apparié (`zero_method="wilcox"`) + delta matched-pairs (`analysis/stats.py`) |
| Évaluation | déterministe sur env fraîche non clippée, wrapper `UTracker` (vraie récompense) |

### Récompense intrinsèque (convention verrouillée, `config.py` / `reward.py`)

    r = r_env + LAMBDA * (U_t - U_{t+1}) / U_max          # mode "chao" (discret)
    r = r_env + LAMBDA * (C_{t+1} - C_t) / U_max          # mode "dense" (continu)

- `U` = Chao-U « remaining work » sur les cibles (`F1²/(2(F2+1))`, variant
  `bias_cap`, capé au nombre de cibles sous-déterminées, plancher 1).
- `C` = somme sur cibles de la variance circulaire des visées → répond à
  **chaque nouvelle visée** (densité ~90 % vs ~0.5 % pour le U discret).
- Positif quand la diversité angulaire / la résolvabilité **augmente**.
- `U_max = num_targets` ; targets jamais observées exclues du cap (le signal
  démarre à plat, comme Project08).

### Propreté du signal (vérifications faites avant la campagne)

- Obs saines : 96 dims/caméra, 28 dims constantes, normalisées |z|≤5, finies.
- Corrélation(récompense clippée, # cibles vues par ≥1 caméra) = **+0.73** ;
  moyenne par état : 0 vue → −0.001, 1 vue → +0.79, 2 vues → +1.52.
- Un MLP supervisé obs→récompense clippée atteint **R² = +0.643** → la
  récompense est bien lisible depuis les obs.
- Métriques : `U_end` sature à 1.0 (non discriminant) ; `C_end` faiblement
  contrôlable ; **`mean_seen`** = moyenne du nb de cibles visibles (métrique
  propre retenue pour la comparaison).

---

## 2. Trois bugs critiques trouvés et corrigés (chacun neutralisait l'apprentissage)

### Bug 1 — Tête de politique saturée (`rl/models.py`)
Moyenne linéaire non bornée + clip → la politique était poussée en coin
(−1,+1) et gelée. **Corrigé** par `TanhNormal` (Gaussienne squasée `tanh`
+ log-prob avec Jacobien) ; `get_distribution`/`act` utilisent `torch.tanh(raw)`.
State dict préservé (checkpoints compatibles).

### Bug 2 — Auto-lambda dégénéré (`environment/reward.py`)
Lambda calibré sur `mean|r_env|` au lieu du **bruit** (std) → λ démesuré
(ordre de 705.5M) et intrinsèque incohérent. **Corrigé** :
- `measure_reward_scale` renvoie `env_std` ;
- `suggest_lambda = target_ratio * noise / unit_p75` ;
- `default_intr_clip = safety * target_ratio * noise` ;
- `--lambda-target-ratio` : défaut 0.25 → **1.0** (`run_mappo.py`).

Le delta unitaire est minuscule en 4v2-9 (p75 ≈ 0.00035) → λ = **46647**,
clip intrinsèque = **58.3**.

### Bug 3 — Clip de gradient commun politique+critique (`rl/mappo.py:update`)
`clip_grad_norm_(params, 0.5)` écrasait la politique (~300×) car les gradients
critique sont ~168 vs ~10.8 politique (**15×**). **Corrigé** par clip séparé :
`clip_grad_norm_(policy.parameters(), 0.5)` puis `critic.parameters()`.
Effet mesuré : valueR2 0 → +0.47, éval (récompense clippée) 427 → 672.

### Correctif complémentaire — `ClipEnvReward`
La récompense MATE = bonus tracking +1/cible visible **plus des pics exogènes
de livraison jusqu'à −500+** qui dominaient les avantages GAE. `ClipEnvReward`
(clip `[−2,2]` pour 2 cibles) retire ces pics du signal d'apprentissage ;
**l'évaluation se fait toujours sur la vraie récompense non clippée**.

---

## 3. Campagne finale `final4` (résultat décisif)

Réalisée **après** les 3 corrections (la campagne antérieure `final100k` — 8
seeds × 100k — avait été lancée avant le fix du Bug 2, avec λ=1410 et clip
intrinsèque 1.76, i.e. intrinsèque négligeable ; elle est donc **supermisée**).

| Paramètre | Valeur |
|---|---|
| Régime | `MATE-4v2-9-v0` (4 caméras, 2 cibles) |
| Seeds | 0..3 (4 seeds) × 2 bras = **8 runs** |
| Étapes | 30 000 / bras |
| Récompense intrinsèque | mode `dense`, fenêtre 25, auto-lambda @ ratio 1.0 → λ=46647.3, clip intr. 58.34 |
| Récompense env | clippée [−2, 2] à l'apprentissage ; vraie à l'éval |
| Éval pendant entraînement | 3 épisodes / 10k étapes (`results/campaign_final4/`) |
| Wall time | ~14.4 steps/s/worker (8 workers = 2 h) |

### Évaluation finale — 10 épisodes déterministes par bras (env fraîche non clippée)

| seed | bras | vraie récompense | mean_seen | C_end |
|---|---|---|---|---|
| 0 | intrinsic | −1325.8 | 0.716 | 1.290 |
| 0 | no_intrinsic | −1270.5 | 0.720 | 1.292 |
| 1 | intrinsic | −1390.2 | 0.701 | 1.276 |
| 1 | no_intrinsic | −1561.2 | 0.577 | 1.215 |
| 2 | intrinsic | −1220.3 | 0.724 | 1.359 |
| 2 | no_intrinsic | −1184.5 | 0.760 | 1.187 |
| 3 | intrinsic | −1619.9 | 0.652 | 1.247 |
| 3 | no_intrinsic | −1717.9 | 0.583 | 1.116 |

| Métrique | intrinsic | no_intrinsic | Δ (intr − noi) | seeds gagnées | Wilcoxon p |
|---|---|---|---|---|---|
| vraie récompense | −1389.05 | −1433.53 | +44.47 | 2/4 | 0.625 |
| mean_seen | +0.70 | +0.66 | +0.04 | 2/4 | 0.625 |
| C_end | +1.29 | +1.20 | +0.09 | 3/4 | 0.250 |

> Note de transparence : une première version du script d'agrégation
> (`eval_final.py`) affichait des moyennes « identiques » (Δ=0.00, p=1.000)
> — un bug d'affichage. Les chiffres ci-dessus sont les agrégations correctes
> recalculées depuis les évaluations par (seed, bras).

---

## 4. Suite : autres estimateurs de richesse (11/08)

### 4.1 Pré-crible des estimateurs (`scripts/prescreen_estimators.py`)

Mesure sur rollouts aléatoires (deltas intrinsèques **unitaires**, signes
identiques à `reward.py`) : densité (% pas non nuls), p75 de |delta|, et
corrélation du delta avec #cibles visibles / r_env clippé.

| estimateur | 4v2 densité | 4v2 p75 | 4v8 densité | 4v8 p75 | 4v8 corr(vis) | 4v8 corr(env) |
|---|---|---|---|---|---|---|
| chao_bias_cap | 0.00 % | — | 2.6 % | 0.25 | 0.009 | −0.026 |
| chao_original | 0.16 % | 0.50 | 3.8 % | 0.31 | −0.006 | 0.005 |
| jackknife | 0.16 % | 0.50 | 3.1 % | 0.125 | 0.017 | −0.006 |
| ace | 0.16 % | 0.50 | 6.5 % | 0.125 | 0.017 | −0.002 |
| dense_cvar | 39 % | 0.006 | 87 % | 0.004 | −0.045 | −0.005 |
| dense_spread | 5 % | 0.050 | 38 % | 0.011 | −0.157 | −0.069 |

Lecture :

- **4v2 : les estimateurs discrets sont quasi morts** — `chao_bias_cap` ne
  déclenche **jamais** (0.00 %), les autres ≈ 0.16 % (≈5 événements/3200 pas).
  La campagne dense de final4 testait donc le seul signal vivant en 4v2, mais
  d'amplitude minuscule.
- **4v8 : tous les signaux sont vivants (2.6-87 %) et décorrélés de la
  visibilité/récompense env (corr ≈ 0)** → l'hypothèse « redondance » du §3
  est **réfutée au niveau pas-à-pas** : le signal de richesse mesure bien
  autre chose que le tracking.
- Le vrai problème de final4 était donc la **faiblesse/rareté du signal**,
  pas sa redondance.

### 4.2 Corrections de calibration nécessaires au mode discret

- `measure_reward_scale` : ajout du p75 des deltas **non nuls**
  (`unit_nonzero_p75`) ; `suggest_lambda` y bascule quand le p75 global est 0
  (signaux clairsemés) — sinon λ ≈ 1e9 (Bug 2 reproduit). Test ajouté :
  `test_suggest_lambda_sparse_signal_falls_back_to_nonzero`.
- Calibration auto désormais sur **l'env d'apprentissage réel**
  (`ClipEnvReward` inclus) : l'ancienne version (env non clippé) sur-calait λ
  de 10-50× (pics de livraison dans l'écart-type). En 4v8 + clip 8 :
  env_std = 2.50 → λ = 10.0 (chaque événement ≈ +2.5 ≈ bruit d'apprentissage).

### 4.3 Campagne `chao4v8` (estimateur canonique : chao discret `bias_cap`)

4v8-9-v0, 4 seeds × 2 bras, 30 000 étapes, `--reward chao --lambda-target-ratio
1.0 --env-reward-clip 8`, éval 3 ép./10k. Évaluation finale : 5 épisodes
déterministes/bras sur env fraîche non clippée
(`results/campaign_chao4v8/eval_final/`).

| seed | bras | vraie récompense | mean_seen | C_end |
|---|---|---|---|---|
| 0 | intrinsic | −7327.4 | 2.544 | 5.406 |
| 0 | no_intrinsic | −7044.0 | 2.630 | 5.065 |
| 1 | intrinsic | −7167.8 | 2.556 | 5.211 |
| 1 | no_intrinsic | −7623.8 | 2.304 | 4.979 |
| 2 | intrinsic | −9131.8 | 1.566 | 5.391 |
| 2 | no_intrinsic | −8959.0 | 1.704 | 4.916 |
| 3 | intrinsic | −8109.4 | 2.308 | 5.128 |
| 3 | no_intrinsic | −8087.4 | 2.253 | 4.889 |

| Métrique | intrinsic | no_intrinsic | Δ (intr − noi) | seeds gagnées | Wilcoxon p |
|---|---|---|---|---|---|
| vraie récompense | −7934.10 | −7928.55 | −5.55 | 1/4 | 0.875 |
| mean_seen | +2.24 | +2.22 | +0.02 | 2/4 | 1.000 |
| C_end | +5.28 | +4.96 | +0.32 | **4/4** | 0.125 |

**Conclusion 4v8 :** toujours aucun gain en récompense ni en couverture, mais
preuve **mécaniste** que l'intrinsèque **modifie bien la politique** : la
diversité angulaire finale `C_end` est supérieure sur **4/4 seeds** (Δ +0.32,
+6.5 %, p=0.125) sans transfert en couverture. L'objectif angulaire
(localisation) est **orthogonal** à l'objectif de tracking récompensé par MATE :
l'intrinsèque ne casse pas la couverture, il l'ajuste en pure perte (le cas
extrême seed 2 : couverture basse 1.57 vs 1.70 mais C_end le plus haut 5.39).

### 4.4 Évaluation localisation (oracle CRLB, `scripts/eval_localization.py`)

Reste une porte de sortie : même sans gain en couverture, la diversité angulaire
apprise (C_end ↑) pourrait se traduire en **localisation** de cibles (c'est
l'objectif du modèle angulaire transposé). On ré-évalue donc les checkpoints des
deux campagnes avec des métriques de localisation bearing-only basées sur l'oracle
CRLB (`QUALITY_SIGMA_BEARING_DEG=1.0°`, seuil `QUALITY_THRESHOLD=100` unités) :
`simult_loc` = fraction de pas où ≥1 cible est bien-localisable (CRLB ≤ seuil),
`loc_peak` = fraction de cibles jamais bien-localisées dans l'épisode, `loc_final`
= idem à l'état final. Éval 3 épisodes déterministes par bras
(`results/campaign_{final4,chao4v8}/eval_localization/`).

**Probe d'échelle CRLB (4v8, 2 ép. aléatoires) :** la carte fait ~700 unités
(caméras jusqu'à |cam|≈679, cibles ≤ 190) ; pour les cibles effectivement vues par
≥2 caméras, les bornes sont p50 ≈ 20, 98 % ≤ 50, 99.4 % ≤ 100 → **le seuil 100 est
bien calibré**. Le goulot est donc la **fréquence des vues simultanées ≥2 caméras**
(~0-2.4 % des pas), pas le seuil.

**`final4` (4v2, 4 seeds) :**

| seed | bras | simult_loc | loc_peak | loc_final |
|---|---|---|---|---|
| 0 | intrinsic | 0.004 | 0.333 | 0.000 |
| 0 | no_intrinsic | 0.000 | 0.000 | 0.000 |
| 1 | intrinsic | 0.000 | 0.000 | 0.000 |
| 1 | no_intrinsic | 0.019 | 0.333 | 0.000 |
| 2 | intrinsic | 0.000 | 0.000 | 0.000 |
| 2 | no_intrinsic | 0.000 | 0.000 | 0.000 |
| 3 | intrinsic | 0.007 | 0.333 | 0.000 |
| 3 | no_intrinsic | 0.006 | 0.333 | 0.000 |

| Métrique | intrinsic | no_intrinsic | Δ | seeds gagnées |
|---|---|---|---|---|
| simult_loc | 0.0028 | 0.0063 | −0.0035 | 2/4 |
| loc_peak | 0.1667 | 0.1667 | 0.0000 | 1/4 |
| loc_final | 0.0000 | 0.0000 | 0.0000 | 0/4 |
| gn_err | — | — | incomplet (aucune cible localisable à l'état final) | |

**`chao4v8` (4v8, 4 seeds) :**

| seed | bras | simult_loc | loc_peak | loc_final |
|---|---|---|---|---|
| 0 | intrinsic | 0.005 | 0.208 | 0.000 |
| 0 | no_intrinsic | 0.001 | 0.167 | 0.000 |
| 1 | intrinsic | 0.003 | 0.125 | 0.000 |
| 1 | no_intrinsic | 0.006 | 0.250 | 0.083 |
| 2 | intrinsic | 0.017 | 0.333 | 0.000 |
| 2 | no_intrinsic | 0.012 | 0.333 | 0.000 |
| 3 | intrinsic | 0.024 | 0.333 | 0.000 |
| 3 | no_intrinsic | 0.021 | 0.333 | 0.000 |

| Métrique | intrinsic | no_intrinsic | Δ | seeds gagnées |
|---|---|---|---|---|
| simult_loc | 0.0125 | 0.0101 | +0.0024 | 3/4 |
| loc_peak | 0.2500 | 0.2708 | −0.0208 | 1/4 |
| loc_final | 0.0000 | 0.0208 | −0.0208 | 0/4 |
| gn_err | — | — | incomplet (aucune cible localisable à l'état final) | |

**Conclusion 4.4 :** la diversité accumulée (C_end ↑) ne se traduit **pas** en
localisation simultanée exploitable. `loc_final` ≈ 0 pour **toutes** les
politiques des deux régimes (jamais ≥2 caméras bien séparées sur une même cible à
l'état final), `simult_loc` est minuscule pour les deux bras et le Δ (3/4 wins en
4v8) est dans le bruit. L'explication est structurelle : le modèle angulaire
accumulé suppose des **cibles statiques** (les vues sont intégrées au cours du
temps), or MATE a des cibles **mobiles** — la richesse accumulée n'est donc pas
transformable en précision de localisation instantanée.

### 4.5 Test de mécanisme : cibles quasi-statiques (`MATE-4v8-9-slow`)

Pourquoi l'intrinsèque n'aide pas sur MATE : le modèle angulaire accumulé
intègre les vues au cours du temps — valide pour des cibles **statiques** — or
MATE a des cibles mobiles. Test de mécanisme : **ralentir les cibles** et
re-mesurer la localisation. Asset `mate/mate/assets/MATE-4v8-9-slow.yaml`
(seule différence : `target.step_size` 20 → 2, déplacement cible mesuré
12.74 → 1.37 u/pas, ~10× plus lent), synchronisé dans `site-packages/mate/assets/`.

**Calibration bloquée (signal mort).** Sur ce régime, le probe aléatoire de
calibration ne fait **jamais** déclencher l'intrinsèque (`unit_p75 = 0` ET
`unit_nonzero_p75 = 0` sur les 4 seeds) → auto-λ dégénéré (jusqu'à 3.8e8).
Une campagne d'apprentissage standard serait donc compromise dès le départ.
Deux réponses :
1. **Robustesse** : `suggest_lambda` gagne un 3ᵉ fallback — si même les unités
   non-nulles sont absentes, λ = target_ratio (échelle neutre finie) au lieu de
   ~1e9. Test verrouillé `test_suggest_lambda_all_zero_units_falls_back_to_neutral`.
2. **Pivot méthodologique** : on n'entraîne pas sur ce régime ; on réutilise les
   checkpoints `chao4v8` (qui diffèrent déjà en C_end) et on les évalue sur
   l'env lent (eval-only, `eval_localization.py --regime MATE-4v8-9-slow-v0`).

**Résultats (5 ép. déterministes, checkpoints chao4v8) :**

| régime env | simult_loc intr/noi | loc_peak intr/noi | loc_final intr/noi | gn_err intr/noi |
|---|---|---|---|---|
| **rapide** (contrôle) | 0.0094 / 0.0083 (2/4) | 0.169 / 0.169 (1/4) | 0.000 / 0.019 (0/4) | 15.6 / 18.1 (n faible) |
| **lent** (mécanisme) | **0.0261 / 0.0061 (+0.020, 3/4)** | 0.138 / 0.106 (1/4) | 0.013 / 0.000 (1/4) | **16.3 / 24.0 (−32 %)** |

Détail seed par seed (env lent) : seed 2 → simult_loc 0.037 vs 0.013, gn_err
14.4 vs 19.5 (212 vs 43 pas bien-localisés) ; seed 3 → simult_loc 0.051 vs
0.011, gn_err 18.2 vs 28.5 (199 vs 8 pas) ; seeds 0-1 ≈ 0 pour les deux bras.

**Conclusion 4.5 :** sur cibles quasi-statiques, les politiques intrinsèques
produisent **4× plus de géométrie de localisation exploitable** (simult_loc) et
une **erreur de triangulation ~32 % plus faible** (gn_err) que les baselines —
le tout **absent sur l'env rapide**. L'intrinsèque Chao-U remplit donc bien son
rôle (créer une diversité angulaire localisable), mais **uniquement quand
l'hypothèse de cible statique tient**. La défaillance sur MATE n'est pas une
faiblesse de l'estimateur mais un **mismatch modèle/domaine** : les cibles
bougent, les relèvements accumulés ne définissent plus un point cohérent.

### 4.6 Oracle séquentiel (fenêtre glissante, compatible cibles mobiles)

Les métriques instantanées (§4.4) ignorent la diversité ACCUMULÉE — le cœur de
l'intrinsèque. Test final : localiser chaque cible depuis son **historique
récent** (fenêtre glissante de 60 pas), le modèle « fenêtré » tolérant au
mouvement :
- `seq_loc` : fraction des pas où une cible est localisable depuis sa fenêtre
  (CRLB de l'historique ≤ seuil) ;
- `seq_err` : erreur Gauss-Newton depuis les relèvements de la fenêtre, vs
  position courante (réservé aux fenêtres multi-vues, ≥ 2 caméras distinctes ;
  init robuste par intersection médiane des paires de relèvements).

`scripts/eval_localization.py --mode seq`, 5 ép. déterministes, checkpoints
`chao4v8` :

| régime env | seq_loc intr/noi | seq_peak intr/noi | seq_err intr/noi |
|---|---|---|---|
| **rapide** (contrôle) | 0.107 / 0.107 (2/4) | 0.200 / 0.200 (0/4) | dominé par le bruit de mouvement (GN 10³-10⁴) |
| **lent** (mécanisme) | **0.083 / 0.043 (+0.040, 3/4)** | 0.144 / 0.106 (1/4) | **39.1 / 52.1 (−25 %)** |

Seed par seed (env lent) : seed 0 → seq_loc 0.044 vs 0.002, seq_err 40.7 vs
77.2 (1580 vs 95 fenêtres localisables) ; seed 2 → 0.160 vs 0.086, 37.9 vs
39.2 ; seed 3 → 0.128 vs 0.083, 38.8 vs 40.0.

**Conclusion 4.6 :** l'oracle séquentiel confirme §4.4 et §4.5 avec une
troisième métrique indépendante : les politiques intrinsèques offrent **~2×
plus de géométrie de localisation exploitable** (seq_loc, 3/4 seeds) et une
**erreur ~25 % plus faible** (seq_err) — mais **uniquement sur cibles
quasi-statiques**. Sur l'env nominal, aucun gain : le mouvement des cibles
invalide la fenêtre accumulée (l'erreur GN explose aux ~10³-10⁴ unités car les
relèvements ne se coupent plus en un point). La conclusion est maintenant
triangulée par trois oracles indépendants (CRLB instantané, GN instantané,
oracle séquentiel fenêtré).

### 4.7 Richesse fenêtrée (`chao4v8win`, W=20) — attaque de la cause racine

La cause racine identifiée (§4.5) : le modèle angulaire **accumulé** suppose
des cibles statiques. Attaque finale : reformuler l'intrinsèque en **richesse à
courte échelle temporelle** — le tracker n'accumule que les relèvements des
`window` dernières étapes (les plus anciens sont évincés), donc U reflète la
diversité angulaire *compatible cibles mobiles*. Implémentation :
`TargetTracker(window=...)` (buffers + timestamps, éviction via `searchsorted`
dans `observe`), propagée `ChaoUReward`/`UTracker` → `--reward-window` dans
`run_mappo.py`/`launch_campaign.py`. Gates 36/36, dont l'éviction et la
**remontée de U après éviction** (la richesse fenêtrée « oublie »).

**Probe de calibration W ∈ {20, 40, 60, 100}** (4v8, rollout aléatoire,
2×500 pas) :

| W (pas) | densité | |ΔU| p95 | |ΔU| non-nuls p75 |
|---|---|---|---|---|
| None (cumulatif) | 2.1 % | 0.000 | 0.250 |
| **20** | **7.8 %** | **0.125** | 0.250 |
| 40 | 4.6 % | 0.000 | 0.250 |
| 60 | 5.0 % | 0.003 | 0.250 |
| 100 | 3.0 % | 0.000 | 0.250 |

Le signal fenêtré est vivant pour tous les W (pas de « fenêtre courte →
signal mort »), et **W=20 maximise la densité** (7.8 %, seul avec une queue p95
non nulle) → c'est le W retenu (la fenêtre la plus courte = la plus valide pour
des cibles mobiles).

**Campagne `chao4v8win`** : MATE-4v8-9-v0, 4 seeds × 2 bras, 30 000 étapes,
`--reward chao --reward-window 20 --lambda-target-ratio 1.0`, éval 3 ép./10k.
λ auto ≈ 211-219 (les fenêtres de calibration ont cette fois capté des pics de
livraison → env_std ≈ 52 ; le ratio cible intrinsèque/bruit reste 1.0 — la
variance de λ est inhérente au protocole, cf. §4.2).

**Évaluation finale** — 5 épisodes déterministes par bras sur env fraîche non
clippée, UTracker fenêtré (`scripts/eval_final.py --window 20`) :

| seed | bras | vraie récompense | mean_seen | C_end fenêtré | C_end cumulatif |
|---|---|---|---|---|---|
| 0 | intrinsic | −8995.2 | 1.547 | 0.012 | 4.872 |
| 0 | no_intrinsic | −6590.4 | 2.833 | 0.208 | 5.032 |
| 1 | intrinsic | −7472.4 | 2.385 | 0.064 | 5.030 |
| 1 | no_intrinsic | −8623.6 | 1.614 | 0.009 | 4.086 |
| 2 | intrinsic | −9211.0 | 1.508 | 0.032 | 5.144 |
| 2 | no_intrinsic | −7196.2 | 2.639 | 0.068 | 4.572 |
| 3 | intrinsic | −7850.2 | 2.225 | 0.364 | 5.158 |
| 3 | no_intrinsic | −8447.2 | 1.928 | 0.215 | 3.717 |

| Métrique | intrinsic | no_intrinsic | Δ | seeds gagnées | Wilcoxon p |
|---|---|---|---|---|---|
| vraie récompense | −8382.20 | −7714.35 | −667.85 | 2/4 | 0.625 |
| mean_seen | +1.92 | +2.25 | −0.34 | 2/4 | 0.625 |
| C_end fenêtré | +0.118 | +0.125 | −0.007 | 2/4 | 1.000 |
| C_end cumulatif | +5.05 | +4.35 | **+0.70** | **3/4** | 0.250 |

Lecture :

- **Aucun gain en couverture** (Δ −668, p=0.625), cohérent avec toutes les
  campagnes précédentes.
- Le patron de diversité **persiste mais en cumulatif** : C_end cumulatif
  +0.70 (3/4, p=0.250) — chaque événement fenêtré laisse une trace qui
  s'accumule sur l'épisode — alors que la diversité **court-terme finale est
  identique** (C_end fenêtré 0.118 vs 0.125, p=1.000). Le signal fenêtré ne
  produit donc pas d'avantage court-terme durable ; il produit un historique de
  diversité aussi orthogonal au tracking que le cumulatif.

**Éval localisation (5 ép. déterministes, `eval_localization.py`) :**

| oracle | métrique | intrinsic | no_intrinsic | Δ | seeds gagnées |
|---|---|---|---|---|---|
| simultané (CRLB) | simult_loc | 0.0105 | 0.0053 | +0.0052 | **4/4** |
| simultané (CRLB) | loc_peak | 0.181 | 0.175 | +0.006 | 1/4 |
| simultané (CRLB) | loc_final | 0.013 | 0.013 | 0.000 | 0/4 |
| séquentiel | seq_loc | 0.117 | 0.083 | +0.033 | **4/4** |
| séquentiel | seq_peak | 0.200 | 0.200 | 0.000 | 0/4 |
| séquentiel | seq_err | 4163 | 2000 | +2163 | 2/4 |

Contrairement à chao4v8 (cumulatif), la politique fenêtrée produit **~2× plus
de géométrie localisable simultanée** (simult_loc 4/4) et **~1.4× plus en
séquentiel** (seq_loc 4/4) — signe mécaniste que le signal fenêtré optimise
exactement ce que mesurent les oracles. **MAIS** : loc_final ≈ 0 partout,
seq_err dominé par le bruit de mouvement (10³-10⁴ u, pire pour l'intrinsèque)
et surtout **aucun transfert en couverture** (p=0.625). La localisation
angulaire — même formulée à courte échelle temporelle — reste un objectif
**orthogonal** au tracking MATE.

**Conclusion 4.7 :** la richesse fenêtrée (la seule formulation attaquant
directement la cause racine « cibles mobiles ») apprend bien la diversité
court-terme (signaux localisables 4/4), mais cela ne se traduit ni en
couverture ni en récompense. La diversité angulaire et le tracking MATE sont
des objectifs structurellement découplés dans ce régime.

### 4.8 Le shaping est potential-based (Ng et al., 1999) : l'échec était en partie prédit

La forme exacte du shaping (`reward.py:ChaoUReward.step`) est

    r_shaped = r_env + λ (U_t − U_{t+1}) / U_max        (sparse)
    r_shaped = r_env + λ (C_{t+1} − C_t) / U_max        (dense)

En posant le potentiel Φ = λ·U/U_max (resp. Φ = −λ·C/U_max pour le dense), on
a r_intr = Φ(s_t) − Φ(s_{t+1}), c'est-à-dire l'opposé de la forme de **shaping
potential-based** de Ng, Harada & Russell (1999) : F(s, a, s′) = γΦ(s′) − Φ(s)
avec γ = 1. Un multiple scalaire d'un shaping potential-based préserve le même
résultat — le théorème s'applique :

> **Théorème (Ng et al., 1999).** Dans un MDP, remplacer la récompense r par
> r + γΦ(s′) − Φ(s) laisse l'ensemble des politiques optimales **inchangé**.

Conséquences directes pour nos trois campagnes :

- **L'intrinsèque ne peut pas modifier la politique optimale asymptotique.**
  Il ne peut agir que sur la **dynamique d'apprentissage** : exploration, crédit
  d'assignation le long de la trajectoire, ou échappement d'optima locaux dans
  un budget d'entraînement fini (30k étapes). C'est exactement ce qui est
  observé — invariance finale (p ≥ 0.625 sur récompense/couverture dans les
  trois régimes) **et** effet d'exploration présent (diversité angulaire
  apprise, C_end ≥ 3/4 seeds, géométrie localisable 4/4 en simultané).
- **Aucune reformulation de la richesse angulaire ne suffira.** Tant que le
  shaping reste une différence de potentiel, l'optimum est invariant. Ceci
  éclaire aussi pourquoi les trois formulations (dense, discret, fenêtré)
  convergent vers le même verdict malgré des signaux très différents.

Validité approximative dans ce cadre (caveats honnêtes) :

- Le théorème suppose un potentiel **fonction de l'état**. Ici U est fonction
  de l'historique d'observations (buffer `TargetTracker`) — en traitant
  l'historique comme un **état d'information augmenté** (ce que la politique
  voit de toute façon via la GRU), l'argument tient approximativement ; la
  non-Markovianité est la même que celle du POMDP MATE brut.
- `U_max` est une constante de référence fixe (`config.py`) → simple mise à
  l'échelle scalaire du potentiel, sans effet.
- Le clip `r_intr_clip` et `ClipEnvReward` rompent la linéarité du shaping :
  le théorème ne s'applique exactement que si les clips ne sont pas saturants
  (λ calibré sur p75, pics rares → saturation rare).

Leçon pour la suite : pour **changer le comportement final** d'un POMDP par
shaping, il faut casser la forme potential-based — potentiel **non stationnaire**
(RND, comptage d'états : le potentiel est estimé *pendant* l'entraînement et
n'est donc pas un potentiel fixe), récompense dépendante de l'historique non
décomposable en différence de potentiel (cohérence multi-vues synchronisée avec
la couverture), ou reshapage de la récompense env elle-même (objectif de
localisation aval). Une intrinsèque potential-based ne peut servir que de
**bootstrap d'exploration** (ex. warm-start intrinsèque puis fine-tune
récompense pure).

### 4.9 Warm-start intrinsèque → fine-tune récompense pure (`warmstart`)

**Hypothèse (§7-1).** Le pré-entraînement intrinsèque (exploration + diversité
apprise, §4.7) devrait donner une **meilleure politique finale** en récompense
pure à budget égal : le fine-tune efface le shaping (le théorème §4.8 ne
s'applique plus pendant la phase 2) tout en héritant de l'exploration.

**Protocole** (8 runs, 4v8, phase-2 = 30k récompense pure, 3 ép./10k, éval 5 ép.
fraîches UTracker W=20) :

- `warm_intrinsic` : init depuis le checkpoint intrinsèque chao4v8win (30k
  intrinsèque) → 30k récompense pure.
- `no_intrinsic` : init depuis le checkpoint no_intrinsic chao4v8win (30k pure)
  → 30k pure. **Contrôle à budget égal** (60k vs 60k) : isole l'effet du
  pré-entraînement intrinsèque, la seule différence étant la récompense des
  premiers 30k.

Implémentation : `run_mappo.py --arm {warm_intrinsic,no_intrinsic}
--init-checkpoint <latest.pt>` (re-base step_count→0, conserve poids +
optimizer), `launch_campaign.py --warm-start --source-tag chao4v8win`
(résolution automatique des checkpoints source). Gates 36/36, smoke OK.

**Résultat — éval finale (5 ép. fraîches, `eval_final.py --arms
warm_intrinsic,no_intrinsic`) :**

| métrique | warm_intrinsic | no_intrinsic (contrôle) | Δ | seeds gagnées | p |
|---|---|---|---|---|---|
| vraie récompense | −8485.7 | −7370.3 | −1115.4 | 2/4 | 0.625 |
| mean_seen | 1.891 | 2.475 | −0.584 | 2/4 | 0.625 |
| U_end fenêtré | 1.350 | 1.725 | −0.375 | 0/4 | 0.250 |
| C_end fenêtré | 0.087 | 0.182 | −0.095 | 3/4 | 0.875 |
| C_end cumulatif | 4.808 | 4.585 | +0.223 | 3/4 | 0.625 |

Lecture :

- **Aucun bénéfice du warm-start.** Le contrôle pur-continué s'améliore bien
  avec le budget supplémentaire (no_intrinsic 30k −7714 → 60k −7370), alors
  que le warm reste **stable autour du comportement intrinsèque** (−8382 au
  départ → −8485 à la fin) : le fine-tune récompense pure n'efface **pas** le
  biais de diversité appris pendant la phase 1.
- C'est la prédiction §4.8 appliquée à l'envers : une intrinsèque
  potential-based peut servir de bootstrap d'exploration, mais ici elle
  **biase durablement l'optimum local** atteint (30k de budget) sans le
  convertir en couverture.

**Éval localisation (5 ép., `eval_localization.py --arms
warm_intrinsic,no_intrinsic`) :**

| oracle | métrique | warm | contrôle | Δ | seeds gagnées |
|---|---|---|---|---|---|
| simultané (CRLB) | simult_loc | 0.0090 | 0.0047 | +0.0043 | **3/4** |
| simultané (CRLB) | loc_peak | 0.188 | 0.131 | +0.056 | 3/4 |
| simultané (CRLB) | loc_final | 0.013 | 0.000 | +0.013 | 2/4 |
| séquentiel | seq_loc | 0.116 | 0.094 | +0.022 | 3/4 |
| séquentiel | seq_err | 5199 | 1605 | +3594 | 3/4 (pire) |

La géométrie localisable **persiste après le fine-tune** (simult_loc ~2×,
seq_loc +0.022, 3/4) — le comportement de diversité est stable, il ne
s'effondre pas sur la récompense pure — mais elle ne se traduit **toujours pas
en couverture** (p=0.625), et l'erreur GN est **pire** pour le warm (seq_err
5199 vs 1605, 3/4) : les configurations « localisables » produites sont de
moins bonne qualité géométrique.

**Conclusion 4.9 :** l'option la plus prometteuse à coût quasi nul
(warm-start) est **réfutée** pour l'objectif de couverture. Le biais
d'exploration intrinsèque persiste mais ne convertit pas, même avec un
fine-tune dédié. Ceci renforce l'orthogonalité structurelle (§5) et pointe
vers les options §7-2 (intrinsèque non-potential, RND) et §7-3 (objectif de
localisation dans la récompense env) comme seules voies capables de changer le
comportement final.

### 4.10 Objectif de localisation dans la récompense env (`loc4v8`)

La suite logique de §4.8 : puisque le shaping intrinsèque est potential-based
(invariant sur l'optimum), tester le **reshapage de la récompense env
elle-même**, qui change réellement la politique optimale. `LocalizationReward`
(`environment/reward.py`) ajoute au signal d'apprentissage la qualité de
localisation instantanée :

    r = r_env + λ · Σ_t q_t,   q_t = clip(QUALITY_THRESHOLD / CRLB_t, 0, 1)

avec `CRLB_t` la borne de Cramér-Rao du target t par `per_target_bounds`
(inf si < 2 observateurs ou Jacobien singulier), `QUALITY_THRESHOLD = 100`.
Le terme n'est **pas** une différence de potentiel (somme par pas de q_t, pas
γΦ(s') − Φ(s)) : l'optimum change en principe. λ est auto-calibré sur l'env
**clippé** (philosophie §4.3 : calibrer sur le signal d'apprentissage) :
λ = 2.50 (= env_std clippé, `env-reward-clip 8`, `loc nonzero_p75 = 1.0`).

Protocole : campagne `loc4v8`, 8 runs (4 seeds × {`loc`, `no_intrinsic`}),
30k étapes, window 20, éval finale sur env fraîche non clippée (5 ép.) +
éval localisation oracle (simult + seq).

Éval finale (vraie récompense / couverture, env fraîche non clippée) :

| métrique             | loc      | no_intr  | Δ        | wins | p     |
|----------------------|----------|----------|----------|------|-------|
| vraie récompense     | −7844.2  | −7928.6  | **+84.4**| 3/4  | 0.625 |
| mean_seen            | 2.248    | 2.146    | +0.102   | 3/4  | 0.625 |
| U_end fenêtré        | 1.700    | 1.800    | −0.100   | 1/4  | 0.875 |
| C_end fenêtré        | 0.116    | 0.182    | −0.067   | 2/4  | 0.625 |
| C_end cumulatif      | 4.999    | 4.962    | +0.037   | 1/4  | 0.875 |

Localisation (oracle CRLB, 5 ép.) :

| métrique   | loc      | no_intr  | Δ        | wins |
|------------|----------|----------|----------|------|
| simult_loc | 0.00798  | 0.00835  | −0.00037 | 1/4  |
| loc_peak   | 0.169    | 0.169    | 0        | 0/4  |
| loc_final  | 0.0125   | 0.0188   | −0.0063  | 1/4  |
| seq_loc    | 0.1065   | 0.1074   | −0.0009  | 1/4  |
| seq_peak   | 0.200    | 0.200    | 0        | 0/4  |
| seq_err    | 5199     | 1929     | +3270    | 1/4 (pire) |

(GN indisponible sur la plupart des seeds faute d'événements localisables :
gn_n > 0 sur 1/4 seeds loc et 1/4 no_intr.)

**Résultat :** même un objectif **non potential-based** dans la récompense env
ne produit **ni gain de couverture ni gain de localisation** en 30k. Le seul
effet est un léger surplus de récompense brute (+84, mean_seen +0.1, 3/4,
p=0.625, non significatif). Le shaping loc est pourtant calibré au niveau du
bruit du signal d'apprentissage (λ = env_std clippé) et l'événement loc
(≥ 2 vues CRLB fini) apparaît ~15 % des pas en politique aléatoire — le signal
est présent mais ne domine pas le bonus de tracking, ou le budget 30k est trop
court pour un objectif géométriquement coûteux et temporellement différé.

**Conclusion 4.10 :** le facteur décisif n'est **pas la forme du shaping**
(potential vs non-potential). La chaîne d'hypothèses était : (a) intrinsèque
potential-based → invariance théorique (§4.8, vérifié) ; (b) reformulation
non-potential dans l'env → changerait le comportement. (b) est testé et ne
suffit pas : le goulot est ailleurs — densité du signal localisable en 4v8,
budget, ou crédit temporel. Caveat λ : pousser λ (×5-10) ferait de la
localisation l'objectif unique et effondrerait la couverture, ce qui ne teste
plus l'effet « accompagnement » recherché.

### 4.11 Random Network Distillation / novelty d'état (`rnd4v8`)

Dernière famille d'intrinsèque non testée (§7-2) et la seule **alignée sur la
couverture** : au lieu de la diversité angulaire, récompenser la **nouveauté
d'état** (Random Network Distillation, Burda et al. 2018). L'état est la pose
de chaque caméra (x/1000, y/1000, theta/180, lu depuis l'état privilégié) ;
la nouveauté est l'erreur de prédiction d'un réseau entraîné (predictor) contre
un réseau cible aléatoire gelé. Le potentiel est **non stationnaire** (le
predictor bouge) → §4.8 ne s'applique pas.

Détails d'implémentation : entraîner le predictor à chaque pas sur la pose
courante **effondre** l'erreur à ~0 (les caméras bougent lentement → le
predictor mémorise la trajectoire) ; la version verrouillée entraîne par
minibatches aléatoires sur un replay buffer (lag) → l'erreur reste ~0.001 sur
les poses familières et ~0.19 sur les poses nouvelles (contraste 200×,
`test_rnd_predictor_converges_on_fixed_states`). λ auto sur env clippé :
**21.5** (p75 de l'erreur 0.116, signal dense 100 % des pas, ≈ bruit env).

Protocole : campagne `rnd4v8`, 8 runs (4 seeds × {`rnd`, `no_intrinsic`}),
30k étapes, clip 8, window 20, éval finale env fraîche (5 ép.) + localisation.

Éval finale (vraie récompense / couverture) :

| métrique         | rnd      | no_intr  | Δ        | wins | p     |
|------------------|----------|----------|----------|------|-------|
| vraie récompense | −8045.3  | −7928.6  | −116.7   | 2/4  | 0.875 |
| mean_seen        | 2.025    | 2.146    | −0.121   | 2/4  | 0.875 |
| U_end fenêtré    | 2.050    | 1.800    | +0.250   | 3/4  | 0.625 |
| C_end fenêtré    | 0.105    | 0.182    | −0.077   | 0/4  | 0.125 |
| C_end cumulatif  | 4.691    | 4.962    | −0.271   | 1/4  | 0.250 |

Localisation (oracle CRLB, 5 ép.) :

| métrique   | rnd      | no_intr  | Δ        | wins |
|------------|----------|----------|----------|------|
| simult_loc | 0.00728  | 0.00835  | −0.00107 | 3/4  |
| loc_peak   | 0.194    | 0.169    | +0.025   | 2/4 (+2 égalités) |
| loc_final  | 0.0063   | 0.0188   | −0.0125  | 0/4  |
| seq_loc    | 0.0978   | 0.1074   | −0.0096  | 2/4  |
| seq_peak   | 0.200    | 0.200    | 0        | 0/4 (égalités) |
| seq_err    | 9539     | 1929     | +7610    | 1/4 (pire) |

(Note simult_loc : moyenne rnd < no_intr, mais victoire **3/4** en apparié —
la moyenne no_intr est gonflée par l'outlier seed 3.)

**Résultat :** la nouveauté d'état **n'améliore pas la couverture** — C_end
fenêtré **0/4**, C_end cumulatif 1/4, vraie récompense −117 (2/4, p=0.875),
mean_seen −0.12. Le bonus de nouveauté fait **errer les caméras** : U_end
fenêtré (diversité angulaire) +0.25 (3/4) et simult_loc 3/4 à niveau très bas
(~0.7 %) progressent légèrement, mais au détriment du tracking (mean_seen
pire). La localisation séquentielle n'est pas améliorée (2/4) et l'erreur GN
est pire (1/4, outliers). En 30k, l'exploration RND ne fournit pas de bootstrap
de couverture exploitable.

**Conclusion 4.11 :** la **dernière famille non testée** est **réfutée** dans
ce régime. Bilan des quatre familles : (1) diversité angulaire potential-based
(§4.8, invariance prédite), (2) warm-start (§4.9), (3) objectif de
localisation non-potential dans l'env (§4.10), (4) novelty d'état RND (§4.11)
— **aucune n'améliore la couverture en 30k**. Le goulot n'est ni la forme du
shaping ni l'alignement du signal (couverture) : c'est la **densité/crédit du
signal et le budget** dans MATE-4v8-9-v0, où le retour de tracking est déjà
optimisé par le contrôle. La seule voie restante crédible est un **budget
beaucoup plus grand** (100k+, accélérateur d'échantillons, §7-11) ou un
changement de régime.

---

### 4.12 Test de régime : découplage diversité/couverture en 8v8 (`chao8v8`)

Hypothèse (§7-4) : si l'échec des quatre familles d'intrinsèque en 4v8 vient
d'une **orthogonalité structurelle** (4 caméras pour 8 cibles → s'écarter pour
voir des cibles différentes est en **tension** avec se regrouper pour de bons
angles), alors dans un régime où cette tension est **relâchée**, la diversité
angulaire devrait enfin se convertir en couverture **et** en localisation. Le
régime 8v8 (`MATE-8v8-9-v0`, 8 caméras pour 8 cibles) donne à chaque cible la
possibilité d'avoir sa propre caméra : la couverture n'exige plus de
sacrifier la géométrie angulaire.

Protocole : campagne `chao8v8`, **16 runs (8 seeds × {`intrinsic`,
`no_intrinsic`})** en deux vagues (0..3 puis extension 4..7), 30k étapes, chao
**fenêtré** W=20 (meilleure forme canonique, §4.7), clip 8, λ auto **10.8**
(|r_env|=2.81, p95 d'unité 0.25), éval finale env fraîche (5 ép., window 20)
+ localisation simult/seq (5 ép.).

Éval finale (vraie récompense / couverture), agrégé n=8 :

| métrique         | intr     | no_intr  | Δ        | wins | p     |
|------------------|----------|----------|----------|------|-------|
| vraie récompense | −7292.8  | −7339.9  | +47.2    | 4/8  | 1.000 |
| mean_seen        | 2.518    | 2.514    | +0.004   | 4/8  | 0.844 |
| U_end fenêtré    | 1.125    | 1.412    | −0.287   | 1/8 (+3 égalités) | 0.250 |
| C_end fenêtré    | 0.402    | 0.184    | **+0.218**| **6/8**| **0.055** |
| C_end cumulatif  | 5.350    | 5.199    | +0.152   | 4/8  | 0.547 |

(Deux vagues indépendantes, mêmes deltas C_end fenêtré : 0..3 → +0.202, 3/4 ;
4..7 → +0.234, 3/4.)

Localisation (oracle CRLB, 5 ép.), agrégé n=8 :

| métrique   | intr    | no_intr  | Δ         | wins |
|------------|---------|----------|-----------|------|
| simult_loc | 0.0218  | 0.0213   | +0.0006   | 3/8  |
| loc_peak   | 0.200    | 0.200    | 0         | 0/8 (plafond) |
| loc_final  | 0.0312  | 0.0125   | **+0.0187**| **6/8** |
| gn_err     | 9.31    | 9.65     | −0.34     | 5/8 (meilleure) |
| seq_loc    | 0.2495  | 0.2416   | +0.0079   | 6/8  |
| seq_peak   | 0.200    | 0.200    | 0         | 0/8 (plafond) |
| seq_err    | 15109   | 11801    | +3308     | 6/8 (lourde queue, pire) |

Détail du C_end fenêtré par seed : s0 0.530 vs 0.133, s1 0.458 vs 0.463,
s2 0.296 vs 0.028, s3 0.268 vs 0.119, s4 0.209 vs 0.400, s5 0.391 vs 0.025,
s6 0.599 vs 0.094, s7 0.220 vs 0.207 — l'intrinsèque gagne **6/8** avec des
marges allant jusqu'à 2-19× (moyenne +0.218, **p=0.055** bilatéral,
≈0.03 unilatéral).

**Résultat (n=8) :** premier signal positif robuste de tout le projet. En 8v8,
l'intrinsèque (1) ne change pas le tracking (mean_seen +0.004, récompense +47,
4/8, neutres), (2) **améliore la couverture de fin d'épisode sur 6/8 seeds**
(C_end fenêtré +0.218, ~2.2× relatif, p=0.055 — la métrique la plus
directement liée à la géométrie encore disponible en fin d'épisode), et
(3) améliore la **localisation de fin d'épisode (loc_final 6/8, +0.0187)** et
la localisation séquentielle (seq_loc 6/8, +0.0079). Le pattern est
**concentré en fin d'épisode** (C_end fenêtré, loc_final) plutôt que dense :
simult_loc est neutre (3/8), les pics sont plafonnés à 0.200 (normalisé,
non discriminatif), et seq_err est bruité (lourde queue, deltas inconsistants
entre les vagues 0..3 et 4..7). L'effet est **modeste mais réel et
répliqué** : la diversité angulaire ne paie pas pendant l'épisode mais laisse
les caméras **verrouillées en paires bien séparées à la fin** (U_end fenêtré
plus bas, C_end +), sans coût sur la récompense.

**Conclusion 4.12 :** l'hypothèse de régime est **partiellement supportée**.
L'orthogonalité diversité/couverture (§5) n'est pas universelle : elle est un
artefact du régime 4v8. En 8v8, la diversité angulaire apprise par
l'intrinsèque se convertit en couverture de fin d'épisode et en localisation
séquentielle (seq_loc **10/12** au n=12), sans coût sur le tracking. Cependant,
l'extension à n=12 dilue le gain C_end fenêtré : **+0.148, 7/12, p=0.092**
(vs +0.218, 6/8, p=0.055 à n=8) — les 4 seeds supplémentaires (8..11) donnent
un gain quasi-nul (+0.009, 1/4). L'effet est donc **réel mais hétérogène** :
certaines configurations de seeds bénéficient de l'intrinsèque (verrouillage
en fin d'épisode) tandis que d'autres non. Le seq_loc reste robuste (10/12),
confirmant que la diversité angulaire améliore la géométrie de formation
même quand elle ne se traduit pas toujours en C_end fenêtré.

---

### 4.13 Déclinaison FOV large 4v8 : marge visuelle vs nombre de caméras
(`chao4v8wide`)

Le gain 8v8 (§4.12) laisse deux explications possibles : (a) **plus de
caméras** (capacité de tracking parallèle : chaque cible peut avoir sa
caméra), ou (b) **plus de marge visuelle par caméra** (le cone plus large
permet de garder la cible en vue tout en changeant d'angle). Le test wide90
isole (b) en gardant 4 caméras pour 8 cibles mais en triplant l'angle de vue.

Régime `MATE-4v8-9-wide90-v0` : copie de `MATE-4v8-9.yaml` avec
`min_viewing_angle: 30 → 90`. Effets sur la dynamique (même
`area_product = angle·range²`) : angle initial ∈ [90,180] et **marge de
rotation ±45°** (vs ±15°), portée ≥ 1061 (vs 612) au lieu de [30,180]/[612,
1500]. obs_dim inchangé (4,126). Campagne `chao4v8wide`, 8 runs (4 seeds ×
{`intrinsic`, `no_intrinsic`}), 30k étapes, chao fenêtré W=20, clip 8,
λ auto **11.8** (|r_env|=3.41). (Lancement et évals par tâche planifiée
Windows — les `Start-Process` étaient tués en fin d'appel outil.)

Éval finale (env fraîche, 5 ép., window 20) :

| métrique         | intr     | no_intr  | Δ        | wins | p     |
|------------------|----------|----------|----------|------|-------|
| vraie récompense | −4936.7  | −4774.6  | −162.2   | 2/4  | 1.000 |
| mean_seen        | 4.071    | 4.201    | −0.130   | 2/4  | 0.875 |
| U_end fenêtré    | 1.525    | 1.925    | −0.400   | 2/4  | 0.625 |
| C_end fenêtré    | 0.713    | 0.886    | −0.173   | 1/4  | 0.625 |
| C_end cumulatif  | 6.032    | 5.449    | **+0.584**| **4/4**| 0.125 |

Localisation (oracle CRLB, 5 ép.) :

| métrique   | intr    | no_intr  | Δ         | wins |
|------------|---------|----------|-----------|------|
| simult_loc | 0.0823  | 0.0932   | −0.0108   | 0/4  |
| loc_peak   | 0.200    | 0.200    | 0         | 0/4 (plafond) |
| loc_final  | 0.0813  | 0.1562   | −0.0750   | 0/4  |
| gn_err     | 16.90   | 16.97    | −0.07     | 2/4  |
| seq_loc    | 0.3857  | 0.3909   | −0.0052   | 2/4  |
| seq_peak   | 0.200    | 0.200    | 0         | 0/8 (plafond) |
| seq_err    | 11927   | 10202    | +1725     | 1/4 (lourde queue) |

**Résultat :** le FOV large **ne réplique pas** le gain 8v8 — l'intrinsèque
perd la couverture de fin d'épisode (C_end fenêtré **1/4**, −0.173) et la
localisation dense (simult_loc **0/4**, loc_final **0/4**). Seul le **C_end
cumulatif** reste positif (4/4, +0.584) : l'intrinsèque accumule la confiance
angulaire sur l'épisode mais ne la convertit ni en géométrie finale ni en
localisation dense. Le tracking reste neutre (récompense 2/4, mean_seen
2/4). Le contraste avec le 8v8 est net : en wide90, le **contrôle exploite
déjà la marge** (C_end fenêtré no_intr 0.886 vs 0.184 en 8v8, mean_seen 4.2
vs 2.5) — il n'y a plus de capacité inexploitée que l'intrinsèque puisse
remplir.

**Conclusion 4.13 :** le mécanisme du gain 8v8 est le **nombre de caméras /
la capacité de tracking parallèle**, pas la marge visuelle par caméra.
L'intrinsèque convertit la diversité en couverture/localisation **là où le
contrôle laisse une capacité géométrique inexploitée** (8v8 : fin d'épisode)
; quand le contrôle sature déjà la géométrie (wide90), l'intrinsèque ne fait
que doper le cumulatif sans gain final. L'orthogonalité 4v8 n'est donc pas
levée par un FOV plus large — seul un excédent de caméras le fait.

### 4.14 Spécificité de l'intrinsèque en 8v8 : RND ne réplique pas le gain
chao (`rnd8v8`)

Le gain 8v8 (§4.12) est établi avec le chao fenêtré W=20. Question
restante : l'effet vient-il du **régime** (n'importe quelle intrinsèque
convertirait dans un régime découplé) ou de la **forme du shaping** (seule
la diversité angulaire convertit) ? Test : RND (novelty d'état,
non-potential, la plus différente mécaniquement, déjà réfutée en 4v8 §4.11)
dans le même régime 8v8, contre le même contrôle no_intrinsic seed-identique.

Protocole : campagne `rnd8v8`, **4 runs = bras `rnd` seul** (seeds 0..3,
30k, clip 8, λ auto **10.53** pour |r_env|=2.70). Contrôle = checkpoints
`no_intrinsic` de `chao8v8` (seeds identiques, entraînement déterministe)
**réutilisés** : ajout de `--skip-no-intrinsic` à `launch_campaign.py` et
copie des dossiers de contrôle dans `campaign_rnd8v8_sXX_no_intrinsic/`
(les évals relisent les checkpoints fraîchement, valeurs identiques à
celles de §4.12). Évals : `eval_final` window 20 (5 ép.) + `eval_localization`
simult/seq (3 ép.).

Éval finale (env fraîche, 5 ép., window 20) :

| métrique         | rnd     | no_intr | Δ        | wins | p     |
|------------------|---------|---------|----------|------|-------|
| vraie récompense | −7292.4 | −7255.6 | −36.8    | 2/4  | 0.625 |
| mean_seen        | 2.546   | 2.590   | −0.045   | 2/4  | 0.875 |
| U_end fenêtré    | 1.775   | 1.300   | **+0.475** | **4/4** | 0.125 |
| C_end fenêtré    | 0.297   | 0.186   | +0.111   | 2/4  | 0.625 |
| C_end cumulatif  | 5.285   | 5.223   | +0.063   | 2/4  | 0.875 |

Localisation (oracle CRLB, 3 ép.) :

| métrique   | rnd    | no_intr | Δ         | wins |
|------------|--------|---------|-----------|------|
| simult_loc | 0.0209 | 0.0198  | +0.0011   | 2/4  |
| loc_final  | 0.0312 | 0.0312  | 0.0000    | 1/4  |
| gn_err     | 11.52  | 7.99    | +3.53     | 2/4  |
| seq_loc    | 0.2638 | 0.2529  | +0.0109   | 2/4  |
| seq_err    | 18833  | 20150   | −1318     | 2/4 (lourde queue) |

**Résultat :** RND est **neutre** en 8v8 — aucun gain répliqué : la
couverture de fin d'épisode est plate (C_end fenêtré **2/4**, +0.111,
p=0.625, vs +0.218/6-8 pour chao) et la localisation ne bouge pas (loc_final
**1/4** vs 6/8, seq_loc **2/4** vs 6/8, gn_err pire). Le contraste le plus
net est le **signe de U_end fenêtré inversé** : RND laisse les caméras avec
**plus** de travail angulaire à la fin (U_end 4/4 plus haut, +0.475, vs
−0.287 pour chao) — la signature « errance » déjà vue en 4v8 (§4.11) — alors
que chao les fait finir verrouillées en paires bien séparées.

**Conclusion 4.14 :** le gain 8v8 est **spécifique à l'objectif de diversité
angulaire fenêtrée**, pas une propriété générale du régime découplé. La
novelty d'état pousse à errer sans convertir en géométrie finale ; seule la
diversité angulaire shaping convertit là où le contrôle laisse une capacité
inexploitée. Le mécanisme du gain 8v8 est donc « diversité angulaire +
excédent de caméras », pas « n'importe quelle exploration + régime
découplé ».

### 4.15 Test croisé 8v8 : cibles quasi-statiques (`MATE-8v8-9-slow`, P1)

Pour lever l'ambiguïté restante (le gain 8v8 vient-il de la cohérence
géométrique statique — prédiction du mismatch §4.5 — ou de la formation
induite par la diversité ?), les checkpoints `chao8v8` ont été ré-évalués sur
l'env lent 8v8 (`MATE-8v8-9-slow`, step_size 2.0, cibles quasi-statiques),
n=16, 5 épisodes deterministes, via 8 tâches planifiées (4 tranches de seeds
× 2 modes). Scripts : `eval_slow_split.cmd`, `agg_p1_slow.py`. JSONs splits
fusionnés : `MATE-8v8-9-slow-v0_{mode}_ep5_{range}.json`.

| métrique   | slow Δ (16 seeds) | wins | 95 % CI (slow) | fast Δ (contrôle) | wins | 95 % CI (fast) |
|------------|-------------------|------|----------------|-------------------|------|----------------|
| simult_loc | +0.0025           | 12/16| [−0.009, +0.012] | +0.0028        | 9/16 | [−0.001, +0.008] |
| seq_loc    | +0.0074           | 12/16| [−0.022, +0.033] | +0.0161        | 12/16| [+0.000, +0.033] |

**Résultat :** le gain 8v8 n'est **pas amplifié** par les cibles lentes —
seq_loc slow (+0.007) est même plus faible que fast (+0.016), et les deux CI
95 % slow incluent zéro. Deux seeds outliers décrochent en slow
(seq Δ −0.097 à s8, −0.138 à s12), variance qui n'existait pas en fast
(±0.03). 

**Interprétation (nuance majeure du mismatch §4.5) :** si le mécanisme était
la cohérence géométrique statique, l'env slow montrerait le gain le plus
fort. Il ne le fait pas : le gain 8v8 est donc **indépendant de la dynamique
cible** — c'est un effet de **formation géométrique** (les caméras se placent
en paires bien séparées, bonus d'exploration dense), et non de localisation
statique. Le mismatch 4v8 (4× en lent, rien en rapide) reste vrai dans le
régime contraint — la diversité n'y produit de la géométrie localisable que
quand les cibles tiennent en place — mais il ne se généralise pas comme
explication du gain 8v8. Conclusion : deux mécanismes distincts selon le
régime, le texte paper mis à jour (Model-Domain Mismatch + abstract +
contributions).

---

## 5. Conclusion et interprétation

- **Aucun gain statistiquement significatif** de l'intrinsèque — ni le dense
  (4v2, `final4`), ni l'estimateur canonique Chao-U discret (4v8, `chao4v8`),
  ni sa version **fenêtrée** compatible cibles mobiles (4v8, `chao4v8win`),
  ni l'objectif de localisation non-potential (4v8, `loc4v8`), ni la novelty
  d'état RND (4v8, `rnd4v8`) — sur la vraie récompense ni sur le `mean_seen`
  (p ≥ 0.625 dans toutes les campagnes) **en régime 4v8**. **Exception 8v8
  (§4.12, n=8)** : dans le régime découplé, l'intrinsèque donne son premier
  gain répliqué sur la couverture de fin d'épisode (C_end fenêtré 6/8,
  +0.218, p=0.055) et la localisation de fin d'épisode (loc_final 6/8).
- **Et c'est théoriquement attendu (§4.8).** Le shaping λ(U_t − U_{t+1})/U_max
  est une **différence de potentiel** — la forme exacte du shaping
  potential-based de Ng et al. (1999) — donc il **laisse l'ensemble des
  politiques optimales invariant**. L'intrinsèque ne peut influencer que la
  dynamique d'apprentissage (exploration, crédit d'assignation), pas l'objectif
  final : ce que les trois campagnes confirment (invariance finale + diversité
  angulaire apprise en exploration). Le résultat négatif n'est pas un échec du
  protocole, mais une **prédiction théorique vérifiée**.
- **Warm-start réfuté (§4.9).** Le pré-entraînement intrinsèque puis un
  fine-tune de 30k sur récompense pure (budget total 60k) ne bat pas le
  contrôle pur-continué à budget égal (vraie récompense −8485 vs −7370,
  p=0.625 ; mean_seen 1.89 vs 2.48). Le contrôle s'améliore avec le budget
  supplémentaire (−7714 → −7370) alors que le warm reste coincé au
  comportement intrinsèque (−8382 → −8485) : le biais de diversité **persiste**
  après le fine-tune (géométrie localisable 3/4) mais ne se convertit jamais en
  couverture. L'intrinsèque potential-based ne fournit pas même un bootstrap
  d'exploration exploitable dans ce régime.
- **L'objectif de localisation dans la récompense env est aussi réfuté
  (§4.10).** `LocalizationReward` (r = r_env + λ·Σ q_t, q_t = clip(100/CRLB_t,
  0, 1), λ = 2.5 calibré sur env clippé) est **non potential-based** : il
  change l'optimum en principe, donc teste la forme du shaping en isolation.
  Résultat en 30k : pas de gain de couverture ni de localisation (simult_loc
  0.00798 vs 0.00835, seq_loc 0.1065 vs 0.1074, 1/4), seul un surplus de
  récompense brute +84 (3/4, p=0.625). **Le goulot n'est donc pas la forme du
  shaping mais la densité/le crédit du signal localisable en 4v8.**
- **La novelty d'état (RND) est réfutée (§4.11).** Le bonus de nouveauté sur
  les poses de caméra (potentiel non stationnaire, λ=21.5) n'améliore pas la
  couverture : C_end fenêtré 0/4, C_end cumulatif 1/4, vraie récompense −117
  (2/4), mean_seen −0.12. Il fait errer les caméras (U_end +0.25, simult_loc
  3/4 à ~0.7 % de niveau) au détriment du tracking. La dernière famille non
  testée et la seule alignée sur la couverture échoue aussi → **le goulot est
  la densité/crédit du signal et le budget en 4v8**, pas la forme du shaping
  ni l'alignement de l'intrinsèque.
- **Hypothèse affinée (l'« orthogonality » remplace la « redondance »).** Le
  pré-crible 4v8 montre que les deltas intrinsèques sont **décorrélés** de la
  visibilité (corr ≈ 0) : le signal ne « recompense pas le tracking ». Mais la
  campagne 4v8 prouve qu'il est appris (C_end > sur 4/4 seeds) sans affecter
  la couverture → la **localisation angulaire est un objectif orthogonal** au
  retour MATE (tracking). C'est pourquoi l'intrinsèque n'apparaît ni comme un
  gain ni comme une nuisance en termes de couverture.
- **L'éval localisation (oracle CRLB, §4.4) sur l'env nominal réfute le
  transfert :** la diversité angulaire accumulée ne donne **aucun** gain en
  localisation simultanée sur des cibles mobiles (loc ≈ 0 pour toutes les
  politiques).
- **MAIS le test de mécanisme (§4.5) prouve que l'estimateur fonctionne :**
  sur des cibles quasi-statiques, les mêmes politiques intrinsèques produisent
  **4× plus de géométrie localisable** (simult_loc +0.020, 3/4 seeds) et une
  **erreur de triangulation 32 % plus faible** (gn_err 16.3 vs 24.0). La
  défaillance sur MATE est donc un **mismatch modèle/domaine** — le modèle
  angulaire accumulé suppose des cibles statiques, les cibles MATE bougent —
  et non une faiblesse de l'estimateur Chao-U.
- **Triangulation par l'oracle séquentiel (§4.6) :** la version fenêtrée
  (tolérante au mouvement) donne le même verdict avec une troisième métrique
  indépendante — ~2× plus de géométrie localisable (seq_loc +0.040, 3/4 seeds)
  et ~25 % d'erreur en moins (seq_err 39 vs 52) sur cibles lentes, aucun gain
  sur cibles mobiles. Trois oracles (CRLB instantané, GN instantané, fenêtre
  glissante) convergent.
- **Attaque de la cause racine par la richesse fenêtrée (§4.7) :** la seule
  formulation compatible cibles mobiles (accumulation sur fenêtre glissante
  W=20) a été implémentée, calibrée (probe W ∈ {20,40,60,100}, W=20 densité
  7.8 %) et entraînée (`chao4v8win`). Résultat : l'intrinsèque fenêtré est
  **appris et mécaniquement présent** — la politique produit ~2× plus de
  géométrie localisable simultanée (simult_loc 0.0105 vs 0.0053, **4/4**) et
  ~1.4× en séquentiel (seq_loc 0.117 vs 0.083, **4/4**) — **mais toujours
  aucun gain en couverture ni en récompense** (Δ −668, p=0.625), et la
  diversité court-terme finale est identique (p=1.000). La cause racine
  n'était donc pas seulement l'échelle temporelle du modèle accumulé : la
  diversité angulaire est **structurellement orthogonale** au tracking MATE.
- **Un transfert réel nécessiterait une reformulation au-delà de la diversité
  angulaire** (ex. objectif directement lié au retour de tracking, comme la
  cohérence multi-vues synchronisée avec la couverture), ou des régimes où la
  diversité angulaire et la visibilité se découplent davantage (FOV plus
  grand, 8v8).
- **L'hypothèse de régime est partiellement supportée par le test 8v8
  (§4.12, n=12).** Dans `MATE-8v8-9-v0` (8 caméras pour 8 cibles), l'intrinsèque
  chao fenêtré améliore le seq_loc sur **10/12 seeds** (+0.022, signal robuste
  de formation localisable), et la couverture de fin d'épisode sur 7/12 seeds
  (C_end fenêtré +0.148). Cependant **p=0.092** — le gain ne franchit pas le
  seuil 0.05 à n=12 (les 4 seeds supplémentaires 8..11 donnent Δ ≈ 0 sur
  C_end fenêtré). L'effet est réel mais **hétérogène** : certaines
  configurations bénéficient du verrouillage en fin d'épisode, d'autres non.
  Le tracking reste neutre (mean_seen +0.040, récompense +116, 5/12).
- **Le mécanisme du gain 8v8 est le nombre de caméras, pas le FOV (§4.13).**
  La déclinaison FOV large 4v8 (`MATE-4v8-9-wide90-v0`, angle 30→90,
  campagne `chao4v8wide`, n=4) **ne réplique pas** le gain : l'intrinsèque y
  perd la couverture de fin d'épisode (C_end fenêtré **1/4**, −0.173) et la
  localisation dense (simult_loc **0/4**, loc_final **0/4**) ; seul le
  cumulatif reste positif (C_end cumulatif **4/4**, +0.584), sans conversion
  finale. En wide90 le contrôle exploite déjà la marge (C_end fenêtré no_intr
  0.886 vs 0.184 en 8v8 ; mean_seen 4.2 vs 2.5) → aucune capacité
  inexploitée. Règle générale affinée : **l'intrinsèque convertit la diversité
  là où le contrôle laisse une capacité géométrique inexploitée** — en 8v8
   (excédent de caméras) c'est la fin d'épisode ; en wide90 rien ne reste.
   L'orthogonalité 4v8 n'est donc pas levée par un FOV plus large.
- **Le gain 8v8 est spécifique à la diversité angulaire, pas au régime
  (§4.14).** RND (novelty d'état, la plus différente mécaniquement) dans le
  même régime 8v8 est **neutre** sur tout : C_end fenêtré 2/4 (+0.111,
  p=0.625, vs +0.218/6-8 pour chao), loc_final 1/4 (vs 6/8), seq_loc 2/4 (vs
  6/8), et surtout **U_end fenêtré 4/4 plus haut** (+0.475, signe inversé vs
  −0.287 pour chao) — RND fait errer les caméras au lieu de les faire finir
  verrouillées en paires séparées. Le mécanisme du gain est donc
  « **diversité angulaire + excédent de caméras** » : ni le régime découplé
  seul, ni la marge visuelle (wide90), ni une exploration générale ne
  suffisent.
- **Limites :** 16 seeds en 8v8 (30k étapes chacun), 4 seeds en wide90 et
  rnd8v8, clip de récompense, une seule topologie FOV, signaux discrets très
  clairsemés en 4v2, éval localisation 3-5 ép. L'effet 8v8 est hétérogène
  (Δ ≈ 0 sur les seeds 8..15) — ni C_end fenêtré (p=0.211) ni seq_loc
  (p=0.130) ne franchit 0.05 à n=16, bien que les CI bootstrap 90 %
  restent > 0.

- **EXTENSION PRÉ-ENGAGÉE à n=16 (seeds 12..15)** : lancée le 17/09/2026
  17:32 (tâche planifiée `chao8v8_s12`, script `chao8v8_s12_all.cmd` →
  `run_chao8v8_s12.cmd` puis `eval_chao8v8_s12.cmd`). Protocole identique aux
  seeds 8..11 (workers=4, 30k étapes/bras, λ auto ratio 1.0, clip env 8,
  window 20, éval 5 ép. final + simult + seq). **Décision prise avant toute
  lecture des résultats : l'extension s'arrête à n=16 ; aucun peek à n=14 ;
  les résultats seront rapportés à n=16 (et n=12 pour comparaison), sans
  cadrage conditionnel au résultat.** Cette phrase est le pré-engagement écrit
  de l'évaluation à n=16.

- **RÉSULTATS n=16 (seeds 12..15 terminés : ALL DONE le 17/09 22:58) :**
  agrégation `agg_robustness.py --tag chao8v8`, n=16, B=20 000, CI 90 % :
  | métrique | n=16 | n=12 (rappels) |
  |---|---|---|
  | seq_loc | **+0.0161, 12/16, p=0.130, CI [+0.003,+0.030] > 0** | +0.022, 10/12, p=0.092 |
  | C_end fenêtré | **+0.106, 8/16, p=0.211, CI [+0.021,+0.193] > 0** | +0.148, 7/12, p=0.092 |
  | C_end cumulatif | +0.103, 11/16, p=0.231, CI [−0.091,+0.293] | +0.159, 8/12, p=0.233 |
  | simult_loc | +0.0028, 9/16, p=0.375, CI [−0.0004,+0.0067] | +0.004, 7/12, p=0.301 |
  | true_rew | +97.5, 8/16, p=0.562 | +116, 5/12, p=1.000 |
  Vague 12-15 (n=4) : C_end fenêtré **−0.020 (1/4)** ; C_end cum **−0.066 (3/4)** ;
  seq_loc **−0.002 (2/4)** ; simult **−0.001 (2/4)** → les seeds 12..15 ne
  montrent plus de gain. **Interprétation honnête :** les effets restent
  positifs en moyenne et leurs CI bootstrap 90 % excluent 0, mais p ne franchit
  pas 0.05 à n=16 ; le papier rapporte les deux n sans cadrage conditionnel.

### Synthèse finale

Le projet a systématiquement testé l'hypothèse « la récompense intrinsèque
de diversité angulaire améliore-t-elle la couverture multi-caméras ? » à
travers 92 runs d'entraînement couvrant 5 familles d'intrinsèque (dense,
chao discret, chao fenêtré, localization objective, RND), 3 régimes (4v2, 4v8,
8v8), des mécanismes de contrôle (FOV large, warm-start), et des tests de
spécificité (RND en 8v8).

**Contributions :**

1. **Vérification empirique de Ng et al. (1999) en MAPPO multi-caméras :** le
   shaped Chao-U (potential-based) laisse l'ensemble des politiques optimales
   invariant — les intrinsèques testées en 4v8 n'affectent ni la récompense ni
   le mean_seen. Le résultat négatif est une prédiction théorique vérifiée.

2. **Découverte du rôle du régime :** l'orthogonalité diversité/couverture
   (§5) n'est pas une loi — elle est un artefact du régime 4v8 (plus de cibles
   que de caméras). En 8v8, le chao fenêtré produit un signal positif (seq_loc
   12/16, C_end fenêtré 8/16, CI 90 % > 0 aux deux n) montrant que la diversité
   angulaire **peut** convertir en couverture quand le contrôle laisse une
   capacité géométrique inexploitée — bien que p ne franchisse pas 0.05 à
   n=16 et que l'effet soit hétérogène (seeds 12..15 neutres). À n=16 les deux
   métriques clés excluent zéro **même au niveau 95 %** (C_end (w) 95 % CI
   [+0.006, +0.211] ; seq_loc 95 % CI [+0.000, +0.033]) ; la table per-seed
   complète (tab:per-seed, appendix) reporte les 16 différences appariées,
   seed = unité d'inférence. La table compacte no-effect 4v8
   (tab:4v8noeffect) confirme par ailleurs Δ+CI 90 % par famille : seul le
   seq_loc du chao fenêtré sort de zéro en 4v8 (contraste du régime), et le
   C_end (w) de RND est significativement négatif (0/4, CI 90 % [-0.111,
   -0.044]) — la novelty d'état dégrade la géométrie de fin d'épisode.

3. **Spécificité de l'objectif :** seuls les intrinsèques de diversité angulaire
   (chao fenêtré) convertissent en 8v8 ; la novelty d'état (RND) et
   l'objectif de localisation direct (loc) ne le font pas. Le mécanisme est
   « diversité angulaire + excédent de caméras » — ni l'exploration générale
   ni la marge visuelle ne suffisent.

4. **Diagnostic de mismatch modèle/domaine :** sur cibles quasi-statiques,
   les mêmes politiques intrinsèques produisent 4× plus de géométrie
   localisable ; sur cibles mobiles MATE, la diversité ne transfère pas en
   localisation simultanée. Le goulot n'est pas l'estimateur mais la
   dynamique du domaine. Le test croisé 8v8 slow (§4.15) nuance : le gain 8v8
   n'est PAS amplifié en quasi-statique — son mécanisme est la formation
   induite par la diversité (indépendant de la dynamique), pas la cohérence
   statique. Le mismatch 4v8 reste un résultat du régime contraint.

**Ouvertures :** l'architecture actuelle sépare la diversité (intrinsèque) de
la couverture (tracking) — un couplage direct (cohérence multi-vues
synchronisée avec la couverture) ou un curriculum 4v2→4v8 pourraient
exploiter le signal là où il est présent (régimes à excédent de caméras).

---

## 6. Reproducibilité

    # Campagne final4 (8 runs = 4 seeds × {intrinsic, no_intrinsic}, dense 4v2)
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag final4 --seeds 0..3 --regime MATE-4v2-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward dense --lambda-target-ratio 1.0 --env-reward-clip 2

    # Campagne chao4v8 (8 runs, chao discret canonique, 4v8)
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao4v8 --seeds 0..3 --regime MATE-4v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8

    # Campagne chao4v8win (8 runs, richesse fenêtrée W=20, 4v8)
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao4v8win --seeds 0..3 --regime MATE-4v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --reward-window 20 --no-baselines

    # Campagne warmstart (8 runs = warm_intrinsic + no_intrinsic, 4v8) :
    #   warm_intrinsic = 30k intrinsèque chao4v8win -> 30k récompense pure
    #   no_intrinsic   = 30k pure chao4v8win -> 30k pure (contrôle budget égal)
    # `--arm warm_intrinsic` + `--init-checkpoint` (run_mappo) re-basent le
    # step_count à 0 : --steps = étapes de la PHASE 2.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag warmstart --seeds 0..3 --regime MATE-4v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --reward-window 20 \
        --warm-start --source-tag chao4v8win --no-baselines

    # Éval finale warmstart (comparaison phase-2, --arms personnalisés)
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag warmstart --regime MATE-4v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20 --arms warm_intrinsic,no_intrinsic

    # Éval localisation warmstart (simult + séquentiel)
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag warmstart --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --arms warm_intrinsic,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag warmstart --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --mode seq --arms warm_intrinsic,no_intrinsic

    # Campagne loc4v8 (8 runs = loc + no_intrinsic, 4v8) : la géométrie
    # localisable devient un OBJECTIF de récompense env (wrapper LocalizationReward,
    # r = r_env + lambda * sum_t q_t, q_t = clip(THRESHOLD/CRLB_t, 0, 1)).
    # lambda auto calibré sur l'env clippé (ici 2.5) ; éval finale sur env
    # fraîche non clippée.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag loc4v8 --seeds 0..3 --regime MATE-4v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward loc --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines

    # Éval finale loc4v8 + localisation (comparaison loc vs no_intrinsic)
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag loc4v8 --regime MATE-4v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20 --arms loc,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag loc4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --arms loc,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag loc4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --mode seq --arms loc,no_intrinsic

    # Campagne rnd4v8 (8 runs = rnd + no_intrinsic, 4v8) : la NOUVEAUTÉ
    # d'état (Random Network Distillation) devient une intrinsèque — potentiel
    # NON stationnaire (le théorème §4.8 ne s'applique pas) qui cible la
    # COUVERTURE directement (poses de caméra inédites = nouvelles zones
    # de l'arène) au lieu de la diversité angulaire. RND prédit l'état pose
    # (x/1000, y/1000, theta/180) ; prédicteur entraîné par minibatches sur
    # un replay buffer (lag, test_rnd_predictor_converges_on_fixed_states) ;
    # lambda auto sur env clippé (≈21). Éval finale sur env fraîche.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag rnd4v8 --seeds 0..3 --regime MATE-4v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward rnd --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines

    # Éval finale rnd4v8 + localisation (comparaison rnd vs no_intrinsic)
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag rnd4v8 --regime MATE-4v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20 --arms rnd,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag rnd4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --arms rnd,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag rnd4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 \
        --mode seq --arms rnd,no_intrinsic

    # Campagne chao8v8 (8 runs = intrinsic + no_intrinsic, 8v8) : test de
    # RÉGIME — 8 caméras pour 8 cibles → la couverture n'est plus en tension
    # avec la diversité angulaire (chaque cible peut avoir sa caméra).
    # Question : l'intrinsèque (chao fenêtré W=20, la meilleure forme
    # canonique) ajoute-t-elle enfin de la localisation SANS perdre de
    # couverture quand l'orthogonalité (§5) est relâchée ?
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao8v8 --seeds 0..3 --regime MATE-8v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines

    # Éval finale chao8v8 + localisation (intrinsic vs no_intrinsic)
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 5 \
        --mode seq

    # Extension chao8v8 : seeds 4..7 (réplication, n=8). Les évals
    # eval_localization fusionnent par (seed, arm) dans le JSON agrégé.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao8v8 --seeds 4..7 --regime MATE-8v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 4..7 \
        --episodes 5 --window 20
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 4..7 --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 4..7 --episodes 5 \
        --mode seq

    # Agrégations appariées (n=8) sans re-rollout
    .venv\Scripts\python.exe scripts\agg_eval_final.py --tag chao8v8
    .venv\Scripts\python.exe scripts\agg_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --episodes 5

    # Déclinaison FOV large 4v8 (mécanisme : marge visuelle vs nombre de
    # caméras). Asset MATE-4v8-9-wide90.yaml = MATE-4v8-9.yaml avec
    # min_viewing_angle 30 -> 90 (marge de rotation ±45° vs ±15°, footprint
    # 3x, portée >= 1061 au lieu de 612). Campagne chao4v8wide (8 runs).
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao4v8wide --seeds 0..3 \
        --regime MATE-4v8-9-wide90-v0 --steps 30000 \
        --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 \
        --episodes 5 --window 20
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 \
        --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8wide --regime MATE-4v8-9-wide90-v0 --seeds 0..3 \
        --episodes 5 --mode seq

    # Spécificité de l'intrinsèque en 8v8 (mécanisme : le gain chao vient-il
    # du régime ou de la forme du shaping ?). RND (novelty, la plus différente
    # mécaniquement) dans le même régime 8v8, bras rnd seul. --skip-no-intrinsic
    # (launch_campaign) n'entraîne QUE le traitement ; le contrôle = checkpoints
    # no_intrinsic de chao8v8 (seed-identiques, déterministes), copiés dans
    # campaign_rnd8v8_sXX_no_intrinsic/ pour que les évals les relisent.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag rnd8v8 --seeds 0..3 --regime MATE-8v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward rnd --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --skip-no-intrinsic --no-baselines
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20 --arms rnd,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 3 \
        --mode simult --arms rnd,no_intrinsic
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag rnd8v8 --regime MATE-8v8-9-v0 --seeds 0..3 --episodes 3 \
        --mode seq --arms rnd,no_intrinsic

    # Extension chao8v8 : seeds 8..11 (n=12 total). Campagne complète
    # (intrinsic + no_intrinsic), evaluation et merge dans JSONs existants.
    .venv\Scripts\python.exe scripts\launch_campaign.py \
        --workers 4 --tag chao8v8 --seeds 8..11 --regime MATE-8v8-9-v0 \
        --steps 30000 --eval-episodes 3 --eval-every 10000 \
        --reward chao --lambda-target-ratio 1.0 --env-reward-clip 8 \
        --reward-window 20 --no-baselines
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 8..11 \
        --episodes 5 --window 20
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 8..11 --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --seeds 8..11 --episodes 5 \
        --mode seq
    # Agrégations n=12 (lisent tous les JSONs eval_final/localisation)
    .venv\Scripts\python.exe scripts\agg_eval_final.py --tag chao8v8
    .venv\Scripts\python.exe scripts\agg_localization.py \
        --tag chao8v8 --regime MATE-8v8-9-v0 --episodes 5





    # Éval finale chao4v8win (5 ép. fraîches, UTracker fenêtré W=20)
    .venv\Scripts\python.exe scripts\eval_final.py \
        --tag chao4v8win --regime MATE-4v8-9-v0 --seeds 0..3 \
        --episodes 5 --window 20

    # Pré-crible des estimateurs (densité / amplitude / corrélation au tracking)
    .venv\Scripts\python.exe scripts\prescreen_estimators.py \
        --regime MATE-4v8-9-v0 --seeds 0..2 --episodes 2 --steps 300

    # Éval localisation (oracle CRLB) sur les checkpoints des deux campagnes
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag final4 --regime MATE-4v2-9-v0 --seeds 0..3 --episodes 5

    # Test de mécanisme : cibles quasi-statiques (asset step_size 20→2)
    # (copier mate/mate/assets/MATE-4v8-9-slow.yaml dans site-packages/mate/assets/)
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8 --regime MATE-4v8-9-slow-v0 --seeds 0..3 --episodes 5

    # Oracle séquentiel (fenêtre glissante, compatible cibles mobiles)
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8 --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 --mode seq
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8 --regime MATE-4v8-9-slow-v0 --seeds 0..3 --episodes 5 --mode seq

    # Éval localisation chao4v8win (simult + séquentiel)
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8win --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5
    .venv\Scripts\python.exe scripts\eval_localization.py \
        --tag chao4v8win --regime MATE-4v8-9-v0 --seeds 0..3 --episodes 5 --mode seq

    # Gates
    .venv\Scripts\python.exe -m pytest tests -q

Artefacts :
- Logs campagne : `results/campaign_final4/worker_*.log`, `results/campaign_chao4v8/worker_*.log`,
  `results/campaign_chao4v8win/worker_*.log`, `results/campaign_warmstart/worker_*.log`,
  `results/campaign_loc4v8/worker_*.log`, `results/campaign_rnd4v8/worker_*.log`
- Runs + checkpoints : `results/campaign_{final4,chao4v8,chao4v8win,warmstart,loc4v8,rnd4v8}_s{00..03}_{intrinsic,no_intrinsic,warm_intrinsic,loc,rnd}/{timestamp}/`
  (config.json, run.log, latest.pt, results.json, DONE)
- Éval finale chao4v8 : `results/campaign_chao4v8/eval_final/` ;
  éval finale chao4v8win : `results/campaign_chao4v8win/eval_final/` ;
  éval finale warmstart : `results/campaign_warmstart/eval_final/` ;
  éval finale loc4v8 : `results/campaign_loc4v8/eval_final/` ;
  éval finale rnd4v8 : `results/campaign_rnd4v8/eval_final/`
- Éval localisation : `results/campaign_{final4,chao4v8,chao4v8win,warmstart,loc4v8,rnd4v8}/eval_localization/`
  (`{regime}_{mode}_ep{episodes}.json`, mode `simult` | `seq`)
- Asset cibles lentes : `mate/mate/assets/MATE-4v8-9-slow.yaml`
- Code : `project09/rl/mappo.py` (clip séparé), `project09/rl/models.py`
  (TanhNormal), `project09/rl/rnd.py` (RND target/predictor),
  `project09/environment/reward.py` (ChaoU/ClipEnvReward/
  calibration non-null + clip d'apprentissage, `window` fenêtré,
  LocalizationReward, RNDIntrinsic), `project09/estimators/continuous.py`
  (`TargetTracker(window=...)`),
  `scripts/{run_mappo,launch_campaign,prescreen_estimators,eval_localization,eval_final}.py`.

Corrections apportées au cours du projet : tête de politique `TanhNormal`
(Bug 1), calibration auto-lambda sur bruit (Bug 2), clip de gradient séparé
politique/critique (Bug 3), `ClipEnvReward` (pics exogènes), fallback
`unit_nonzero_p75` pour les signaux clairsemés + calibration sur l'env clippé
(réparations requises pour le mode discret en 4v8), et fallback neutre
`λ = target_ratio` quand même les unités non-nulles sont absentes (cibles
quasi-statiques, §4.5).

---

## 7. Suites potentielles (consignées, non testées)

Toutes les idées discutées sont listées ici avec leur statut. Classées par
rapport espérance/coût estimé.

1. **[RÉFUTÉ — §4.9] Warm-start intrinsèque → fine-tune récompense pure.**
   Testé (campagne `warmstart`, 8 runs) : à budget égal (60k), le
   pré-entraînement intrinsèque ne bat pas le contrôle pur-continué
   (récompense −8485 vs −7370, p=0.625) ; le biais de diversité persiste après
   le fine-tune sans se convertir en couverture.
2. **[RÉFUTÉ — §4.11] Intrinsèque non-potential : RND ou comptage d'états.**
   Potentiel non-stationnaire estimé pendant l'entraînement → casse le
   théorème (§4.8). Implémenté (`project09/rl/rnd.py`, `RNDIntrinsic`,
   replay buffer + lag, λ auto 21.5, campagne `rnd4v8`) et testé : pas de gain
   de couverture (C_end fenêtré 0/4, vraie récompense −117), les caméras
   errent (U_end +0.25, simult_loc 3/4 à ~0.7 %) au détriment du tracking.
   La seule intrinsèque alignée sur la couverture échoue aussi → le goulot est
   la densité/crédit du signal et le budget en 4v8, pas la forme du shaping.
3. **[RÉFUTÉ — §4.10] Objectif de localisation dans la récompense env.**
   Récompenser directement la géométrie localisable (qualité CRLB, cohérence
   multi-vues synchronisée avec la couverture). Seul moyen de changer l'optimum
   vers la localisation, mais change la question de recherche. Implémenté
   (`LocalizationReward`, λ auto 2.5, `campagne loc4v8`) et testé : pas de gain
   de couverture ni de localisation en 30k (simult_loc 0.00798 vs 0.00835),
   seul +84 de récompense brute (3/4, p=0.625). La forme non-potential ne suffit
   pas — le goulot est la densité du signal localisable en 4v8.
4. **[PARTIELLEMENT SUPPORTÉ — §4.12 (n=12) ; mécanisme §4.13, spécificité
   §4.14] Régimes découplant diversité/couverture : FOV large ou 8v8.**
   Campagne `chao8v8` (MATE-8v8-9-v0, 8 caméras / 8 cibles, **24 runs =
   intrinsic+no_intrinsic × 12 seeds**, chao fenêtré W=20, clip 8, 30k) :
   seq_loc **10/12** (+0.022, signal robuste de formation localisable), C_end
   fenêtré 7/12 (+0.148, **p=0.092** — ne franchit pas 0.05), tracking
   neutre. Les 4 seeds supplémentaires (8..11) donnent Δ ≈ 0 sur C_end,
   diluant le gain n=8 (p=0.055). Effet réel mais hétérogène. **Déclinaison
   FOV large (§4.13) réfutée** (C_end fenêtré 1/4, loc_final 0/4).
   **Spécificité (§4.14) confirmée** : RND dans le même régime 8v8 est
   neutre (C_end fenêtré 2/4, U_end fenêtré 4/4 plus haut — signature
   errance) → le gain est « diversité angulaire + excédent de caméras », pas
   « n'importe quelle exploration + régime découplé ».
5. **[CONFIRMÉ — §4.XX GAE] Diagnostic d'attribution GAE.** Offline
   (`scripts/diagnose_gae.py`, 8 checkpoints = seeds 0/5/9/15 ×
   intrinsic/no_intrinsic, horizon 350, politique + critique finaux) :
   décomposition exacte de l'avantage par flux de récompense (GAE linéaire
   en récompenses à critique fixe). Résultat : le shaping est **écrasé par le
   bruit env** dans le crédit appris. `var_share_intr` ≈ 0.00005 (max 0.00013)
   sur la variance de l'avantage, `corr(r_intr, adv)` ≈ 0 (−0.037…+0.047),
   `std contrib_env / contrib_intr` ≈ **147×**. Cause : le signal chao
   discret ne tire que ~8 % des pas (p75 des |r_intr| = 0) et, quand il tire,
   |r_intr| max ≈ 4 vs |r_env| std ≈ 53 (λ=10.8 calibré sur le p75 des
   deltas non-nuls, pas sur la rareté de déclenchement). Conclusion :
   à budget 30k, le shaping potential-based ne génère quasiment aucun
   gradient d'exploration supplémentaire — le goulot est la **densité de
   déclenchement**, pas l'amplitude.
6. **Annealing de λ.** Décroître λ au cours de l'entraînement (exploration
   forte puis recentrage sur la récompense pure).
7. **Diversité conditionnée au tracking.** Récompenser la diversité seulement
   quand la cible est déjà vue (découple visibilité et géométrie).
8. **Shaping de formation d'équipe.** Bonus pour la cohérence de la formation
   de caméras (couplage diversité/couverture).
9. **Currículo 4v2 → 4v8.** Entraîner sur le régime facile puis fine-tuner sur
   le difficile.
10. **Tête auxiliaire prédictive.** Tête de prédiction de cible (self-supervisé)
    en complément de la politique.
11. **Accélérateur d'échantillons.** Réduire le coût par run (env/RL
    vectorisés) pour permettre des budgets 100k+.
12. **Clip env 4v8 plus agressif.** Baisser le clip de la récompense
    d'apprentissage (ex. 2 au lieu de 8) pour vérifier le diagnostic GAE sans
    nouvelle implémentation.

Leçons transversales déjà établies : toute intrinsèque **potential-based**
(§4.8) ne peut servir que de bootstrap d'exploration ; pour changer le
comportement final il faut casser la forme (potentiel non stationnaire,
dépendance historique, ou reshapage de la récompense env elle-même) — mais
§4.10 montre que ce reshapage non-potential ne suffit pas non plus en 30k,
et §4.11 que même la novelty d'état alignée couverture (RND) échoue : le
goulot est la **densité/crédit du signal en 4v8**, pas la forme du shaping.
