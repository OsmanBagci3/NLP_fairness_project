# 🔬 Analyse Croisée : Uncertainty Quantification × NLP Fairness Project

> **Théorie source** : Cours UQ en Deep Learning — Gianni Franchi, U2IS, ENSTA Paris (2026)
> **Projet** : Classification de professions médicales — détection/mitigation des biais de genre
> **Date** : 2026-04-23

---

## 1. CARTOGRAPHIE — Mon projet vs. Théorie UQ

### 1A. Ce que j'ai DÉJÀ utilisé de la théorie UQ

---

#### 🟢 Random Forest = Bagging Ensemble

**Théorie UQ — Méthodes d'ensemble** : Les méthodes d'ensemble combinent plusieurs modèles pour améliorer la robustesse et capturer l'incertitude. Le Bagging entraîne plusieurs instances du même modèle sur des sous-ensembles bootstrap et agrège par vote.

**Ce que j'ai fait** : Mon RF avec `n_estimators=100` (NB01, NB04) est un ensemble Bagging de 100 arbres de décision. Chaque arbre est entraîné sur un tirage bootstrap du dataset. La sortie `rf.predict_proba(x)` est :

$$P_{\text{RF}}(Y=k|x) = \frac{1}{100}\sum_{j=1}^{100} \mathbb{1}[h_j(x) = k]$$

C'est exactement la formule d'ensemble :
$$P(y^*|x^*) = \frac{1}{N_{\text{model}}}\sum_{j=1}^{N_{\text{model}}} P(y^*|\omega_j, x^*)$$

La **variance entre arbres** est une mesure d'incertitude épistémique que je n'exploite pas encore formellement.

---

#### 🟢 Dropout dans FairMLP = Ensemble implicite (→ MC Dropout potentiel)

**Théorie UQ — MC Dropout** : Le MC Dropout simule un ensemble en appliquant le dropout pendant la phase d'inférence :
$$P(y^*|x^*) = \frac{1}{N_{\text{model}}}\sum_{j=1}^{N_{\text{model}}} P(y^*|\omega(t^*) \odot b^j, x^*)$$

**Ce que j'ai fait** : Mon FairMLP (NB10) utilise `Dropout(p=0.3)` entre chaque couche cachée, mais **uniquement en mode régularisation** — `model.eval()` le désactive à l'inférence. Je suis à une ligne de code (`model.train()`) du MC Dropout.

---

#### 🟢 Bootstrap Confidence Intervals = Incertitude paramétrique

**Théorie UQ — Incertitude épistémique** : L'incertitude épistémique provient des limites du modèle et du manque de connaissances (données limitées). Elle peut être réduite en ajoutant des données.

**Ce que j'ai fait** : Bootstrap CI (2000 itérations, NB08) sur mes métriques fairness :

| Métrique | Valeur | CI 95% | Interprétation UQ |
|---|---|---|---|
| accuracy | 0.899 | [0.880, 0.917] | Incertitude paramétrique modérée |
| ∆DP | 0.175 | [0.154, 0.196] | 🚨 Biais robuste (CI exclut 0) |
| ∆EO | 0.181 | [0.112, 0.251] | 🚨 Biais robuste, mais CI large = incertitude élevée |
| ∆EqOdds | 0.183 | [0.115, 0.253] | 🚨 Idem |

Le CI de ∆EO est 3× plus large (0.139) que celui de l'accuracy (0.037) → le biais est plus **épistémiquement incertain** que la performance elle-même.

---

#### 🟢 Courbes de calibration = Diagnostic d'overconfidence

**Théorie UQ — ECE et surconfiance** : Quantifier l'incertitude prévient les prédictions overconfident qui peuvent mener à des décisions biaisées ou injustes. L'ECE mesure l'alignement entre confiance prédite et accuracy réelle :
$$\text{ECE} = \sum_{i=1}^{m} \frac{|B_i|}{N} \cdot |\text{accuracy}(B_i) - \text{confidence}(B_i)|$$

**Ce que j'ai fait** : NB07 — courbes de calibration par genre, histogrammes de `max_proba`, comparaison correct vs incorrect. Le diagnostic est **qualitatif** (graphiques) mais pas encore **quantitatif** (score ECE).

---

#### 🟢 Distribution shift temporel 2017→2018

**Théorie UQ — Distribution Shifts** : Décalage entre données d'entraînement et de déploiement. Il peut être sensoriel (changement de distribution de X) ou sémantique (nouvelles classes ou changement de P(Y)).

**Ce que j'ai fait** : NB13 — stabilité temporelle :
- 2017 (N=344) : acc=86.6%, ∆EO=14.7%
- 2018 (N=656) : acc=91.6%, ∆EO=18.9%

C'est un **shift mixte** : le vocabulaire des biographies évolue (sensoriel) et la distribution genre×profession peut changer (sémantique). Le cadre UQ fournit le vocabulaire formel pour ce que j'observe empiriquement.

---

#### 🟢 Data Imbalance = Source d'incertitude

**Théorie UQ — Sources d'incertitude** : L'incertitude provient aussi de données d'entraînement limitées — les datasets déséquilibrés produisent des modèles avec des prédictions incertaines pour les classes sous-représentées.

**Ce que j'ai fait** : Tout le projet. Le déséquilibre genre×profession (peu de femmes chirurgiennes, peu d'hommes infirmiers) est la source fondamentale des biais. Les ∆EO élevés (nurse=35%, surgeon=44.5%) reflètent directement l'incertitude du modèle sur les sous-groupes minoritaires.

---

#### 🟢 Sensibilité hyperparamètres + taille échantillon

**Théorie UQ — Incertitude épistémique** : L'incertitude épistémique peut être réduite avec plus de données ou de meilleurs modèles.

**Ce que j'ai fait** :
- Grid search RF 27 configs (NB04) : le composite score varie de 0.726 à 0.904
- λ-sweep fairness loss 10 configs (NB10)
- Sensibilité taille N=500→8000 (NB13) : les gaps fairness augmentent sous N=2000 → confirme que l'incertitude épistémique diminue avec plus de données

---

#### 🟢 Cross-validation 5-fold

**Ce que j'ai fait** : NB13 — 5-Fold CV :
- accuracy : 0.886 ± 0.008 (stable)
- ∆EO : 0.193 ± 0.027 (3× plus variable)

La **variance** de ∆EO entre folds est une estimation directe de l'incertitude épistémique sur le biais. C'est un résultat important : on est plus incertain sur le niveau de biais que sur la performance.

---

### 1B. Ce que je POURRAIS appliquer de nouveau

---

#### 🟡 MC Dropout — Victoire rapide

**Théorie** : Garder le dropout actif à l'inférence, faire $T$ forward passes. Chaque passe masque des neurones différents → distribution de prédictions.

**Pourquoi c'est pertinent** : Permet de mesurer si le modèle est **plus épistémiquement incertain** sur les femmes chirurgiennes que sur les hommes chirurgiens — un type de biais invisible à ∆EO.

---

#### 🟡 Deep Ensembles

**Théorie** : Entraîner $M$ DNNs avec initialisations aléatoires différentes et backpropagation non-déterministe :
$$P(y^*|x^*) = \frac{1}{N_{\text{model}}}\sum_{j=1}^{N_{\text{model}}} P(y^*|\omega_j(t^*), x^*)$$

**Pourquoi c'est pertinent** : 5 FairMLPs (même λ=0.1, seeds différentes) → le **disagreement** par genre×profession est une mesure de biais épistémique.

---

#### 🟡 ECE par genre — Biais de calibration

**Théorie** : L'ECE mesure la miscalibration globale. Appliqué par genre, il détecte un type de biais invisible à ∆EO.

**Pourquoi c'est pertinent** : Un modèle peut avoir ∆EO≈0 mais ∆ECE élevé — quand il dit "95% confiant" pour une femme, il a raison 85% du temps, vs 97% pour un homme. Injustice de calibration.

---

#### 🟡 Weak Ensembles (BatchEnsemble, MIMO, Packed-Ensembles)

**Théorie** : Alternatives rapides aux Deep Ensembles. BatchEnsemble utilise des poids rapides de rang 1 ($W_j = W \cdot r_j s_j^T$). MIMO utilise un seul modèle avec M entrées/sorties. Packed-Ensembles utilise des grouped convolutions.

**Pourquoi c'est pertinent** : Si je scale vers du fine-tuning de transformers, ces méthodes donneraient l'UQ à coût réduit.

---

#### 🟡 BNN Post-Hoc (ABNN)

**Théorie** : Convertir un DNN existant en BNN en remplaçant les couches de normalisation par des Bayesian Normalization Layers :
$$\text{BNL}(h_j) = \frac{h_j - \hat{\mu}_j}{\hat{\sigma}_j} \times \gamma_j(1 + \epsilon_j) + \beta_j, \quad \epsilon_j \sim \mathcal{N}(0, I)$$

**Pourquoi c'est pertinent** : Ajouter des LayerNorm dans mon FairMLP, puis les convertir en BNL, donnerait une incertitude bayésienne complète sans ré-entraînement lourd.

---

#### 🟡 OOD Detection via incertitude

**Théorie** : AUROC, AUPR, FPR95 évaluent la capacité du modèle à détecter des données hors distribution, liées à l'incertitude épistémique.

**Pourquoi c'est pertinent** : Traiter les sous-populations mal servies (Female×surgeon) comme "OOD interne" et appliquer ces métriques = pont entre OOD detection et audit de biais.

---

## 2. FONDEMENTS MATHÉMATIQUES + EXEMPLES NUMÉRIQUES

### 2.1 Entropie Prédictive — Mesure de confiance d'un DNN unique

**Formulation** :

Pour un DNN unique avec $K=5$ classes (dentist, nurse, physician, psychologist, surgeon) :

$$H(P(Y|x, \omega)) = -\sum_{k=1}^{K} P(Y=k|x, \omega) \log P(Y=k|x, \omega)$$

- $H_{\min} = 0$ nats → certitude totale (toute la masse sur une classe)
- $H_{\max} = \ln(5) = 1.609$ nats → distribution uniforme (incertitude maximale)

**L'entropie mesure l'incertitude aléatoire** pour un DNN unique. Pour un ensemble, elle capture l'incertitude totale (aléatoire + épistémique).

**Exemple numérique — Biographie de nurse Female** :

Mon FairMLP (λ=0.1 eo) produit le softmax suivant :

| Classe | dentist | nurse | physician | psychologist | surgeon |
|---|---|---|---|---|---|
| $P(Y=k|x)$ | 0.02 | 0.88 | 0.03 | 0.04 | 0.03 |

Calcul terme par terme :

| k | $P_k$ | $\ln P_k$ | $-P_k \ln P_k$ |
|---|---|---|---|
| dentist | 0.02 | −3.912 | 0.078 |
| nurse | 0.88 | −0.128 | 0.112 |
| physician | 0.03 | −3.507 | 0.105 |
| psychologist | 0.04 | −3.219 | 0.129 |
| surgeon | 0.03 | −3.507 | 0.105 |

$$H = 0.078 + 0.112 + 0.105 + 0.129 + 0.105 = \mathbf{0.529} \text{ nats}$$

Confiance normalisée : $1 - H/H_{\max} = 1 - 0.529/1.609 = \mathbf{67.1\%}$

**Même biographie, profil masculin** — le modèle hésite davantage :

| Classe | dentist | nurse | physician | psychologist | surgeon |
|---|---|---|---|---|---|
| $P(Y=k|x')$ | 0.03 | 0.55 | 0.15 | 0.07 | 0.20 |

| k | $P_k$ | $\ln P_k$ | $-P_k \ln P_k$ |
|---|---|---|---|
| dentist | 0.03 | −3.507 | 0.105 |
| nurse | 0.55 | −0.598 | 0.329 |
| physician | 0.15 | −1.897 | 0.285 |
| psychologist | 0.07 | −2.659 | 0.186 |
| surgeon | 0.20 | −1.609 | 0.322 |

$$H' = 0.105 + 0.329 + 0.285 + 0.186 + 0.322 = \mathbf{1.227} \text{ nats}$$

Confiance normalisée : $1 - 1.227/1.609 = \mathbf{23.7\%}$

**Nouveau diagnostic de biais — ∆H** :

$$\Delta H_c = \mathbb{E}_{x \in \text{Female}, y=c}[H(P(Y|x))] - \mathbb{E}_{x \in \text{Male}, y=c}[H(P(Y|x))]$$

Ici, pour la classe nurse : $\Delta H_{\text{nurse}} = 0.529 - 1.227 = -0.698$.

Le signe négatif signifie que le modèle est **plus confiant** sur les nurses femmes que sur les nurses hommes. Cet écart de confiance est une forme de biais **non capturée par ∆EO**, car ∆EO ne mesure que les erreurs binaires (correct/incorrect), pas la gradation de confiance.

> **Lien avec la théorie UQ** : La surconfiance (overconfidence) peut mener à des décisions biaisées ou injustes. Quantifier cette surconfiance par genre via l'entropie est un cas concret de ce que le cours identifie comme prérequis éthique.

---

### 2.2 MC Dropout — Décomposition Aléatoire/Épistémique

**Formulation** :

À l'inférence, $T$ forward passes avec dropout activé :

$$\bar{P}(Y=k|x) = \frac{1}{T} \sum_{t=1}^{T} P(Y=k|x, \omega \odot b_t)$$

où $b_t \sim \text{Bernoulli}(1-p)^{|\omega|}$ est le masque dropout au pass $t$.

**Décomposition clé** (lien direct avec la taxonomie aléatoire/épistémique du cours) :

$$\underbrace{H[\bar{P}]}_{\text{incertitude totale}} = \underbrace{\frac{1}{T}\sum_t H[P_t]}_{\text{aléatoire (irréductible)}} + \underbrace{I(Y|x)}_{\text{épistémique (réductible)}}$$

où l'**information mutuelle** est :

$$I(Y|x) = H[\bar{P}] - \frac{1}{T}\sum_t H[P_t]$$

**Exemple numérique — T=3 passes, biographie surgeon Male** :

| Pass $t$ | dentist | nurse | physician | psychologist | surgeon |
|---|---|---|---|---|---|
| $t=1$ | 0.05 | 0.02 | 0.10 | 0.03 | **0.80** |
| $t=2$ | 0.08 | 0.04 | 0.25 | 0.08 | **0.55** |
| $t=3$ | 0.03 | 0.01 | 0.08 | 0.02 | **0.86** |

**Étape 1 — Prédiction ensemble** :

$$\bar{P} = \left[\frac{0.05+0.08+0.03}{3},\ \frac{0.02+0.04+0.01}{3},\ \frac{0.10+0.25+0.08}{3},\ \frac{0.03+0.08+0.02}{3},\ \frac{0.80+0.55+0.86}{3}\right]$$
$$\bar{P} = [0.053,\ 0.023,\ 0.143,\ 0.043,\ 0.737]$$

**Étape 2 — Incertitude totale** $H[\bar{P}]$ :

| k | $\bar{P}_k$ | $-\bar{P}_k \ln \bar{P}_k$ |
|---|---|---|
| dentist | 0.053 | 0.156 |
| nurse | 0.023 | 0.087 |
| physician | 0.143 | 0.278 |
| psychologist | 0.043 | 0.135 |
| surgeon | 0.737 | 0.225 |

$$H[\bar{P}] = 0.156 + 0.087 + 0.278 + 0.135 + 0.225 = \mathbf{0.881} \text{ nats}$$

**Étape 3 — Entropies individuelles** (incertitude aléatoire de chaque passe) :

$H[P_1] = -(0.05 \ln 0.05 + 0.02 \ln 0.02 + 0.10 \ln 0.10 + 0.03 \ln 0.03 + 0.80 \ln 0.80)$
$= 0.150 + 0.078 + 0.230 + 0.105 + 0.179 = 0.742$

$H[P_2] = -(0.08 \ln 0.08 + 0.04 \ln 0.04 + 0.25 \ln 0.25 + 0.08 \ln 0.08 + 0.55 \ln 0.55)$
$= 0.202 + 0.129 + 0.347 + 0.202 + 0.329 = 1.209$

$H[P_3] = -(0.03 \ln 0.03 + 0.01 \ln 0.01 + 0.08 \ln 0.08 + 0.02 \ln 0.02 + 0.86 \ln 0.86)$
$= 0.105 + 0.046 + 0.202 + 0.078 + 0.130 = 0.561$

**Étape 4 — Incertitude aléatoire moyenne** :

$$\bar{H} = \frac{0.742 + 1.209 + 0.561}{3} = \mathbf{0.837} \text{ nats}$$

**Étape 5 — Information Mutuelle** (incertitude épistémique) :

$$I(Y|x) = H[\bar{P}] - \bar{H} = 0.881 - 0.837 = \mathbf{0.044} \text{ nats}$$

**Interprétation** :

| Composante | Valeur | % du total | Type (cours) | Signification pour le projet |
|---|---|---|---|---|
| Aléatoire $\bar{H}$ | 0.837 | 95% | Irréductible | Le texte lui-même est ambigu (surgeon vs physician) — plus de données ne résoudra pas ça |
| Épistémique $I$ | 0.044 | 5% | Réductible | Le modèle est relativement stable — ses poids sont bien ajustés pour ce profil |

**Application fairness** : Si on refait le même calcul pour une **surgeon Female** et qu'on obtient $I = 0.180$ (MI 4× plus élevée) :
- Le modèle est **structurellement instable** sur les profils féminins chirurgiens
- Ses poids ne sont pas bien ajustés pour ce sous-groupe → **incertitude épistémique genrée**
- C'est un biais invisible à ∆EO qui ne se révèle qu'avec MC Dropout

On peut définir :
$$\Delta \text{MI}_c = |\mathbb{E}_{\text{Female}, y=c}[I(Y|x)] - \mathbb{E}_{\text{Male}, y=c}[I(Y|x)]|$$

Un $\Delta\text{MI}$ élevé pour nurse ou surgeon confirmerait que le biais de genre est en partie **épistémique** (réductible avec plus de données de ces sous-groupes).

---

### 2.3 Deep Ensembles

**Formulation** :

$$P(y^*|x^*) = \frac{1}{M} \sum_{j=1}^{M} P(y^*|\omega_j(t^*), x^*)$$

Chaque $\omega_j$ est entraîné avec un seed différent. Les sources de diversité : initialisation aléatoire, SGD stochastique, backpropagation non-déterministe.

**Mesures dérivées** :

Disagreement (taux de désaccord) :
$$D(x) = 1 - \frac{1}{M}\sum_{j=1}^{M} \mathbb{1}[\hat{y}_j = \hat{y}_{\text{ensemble}}]$$

Variance inter-modèles sur la probabilité d'une classe :
$$\text{Var}_c(x) = \frac{1}{M}\sum_{j=1}^{M} \left(P_j(Y=c|x) - \bar{P}(Y=c|x)\right)^2$$

**Exemple numérique — M=5 FairMLPs, biographie nurse Male** :

| Modèle $j$ | Prédiction | P(nurse) |
|---|---|---|
| $\omega_1$ (seed=42) | nurse | 0.72 |
| $\omega_2$ (seed=123) | physician | 0.31 |
| $\omega_3$ (seed=456) | nurse | 0.68 |
| $\omega_4$ (seed=789) | nurse | 0.65 |
| $\omega_5$ (seed=1024) | surgeon | 0.22 |

- Prédiction ensemble : **nurse** (3/5 votes)
- Disagreement : $D = 1 - 3/5 = 0.40$ → 40% de désaccord
- $\bar{P}(\text{nurse}) = (0.72 + 0.31 + 0.68 + 0.65 + 0.22)/5 = 0.516$
- $\text{Var}_{\text{nurse}} = \frac{1}{5}[(0.72-0.516)^2 + (0.31-0.516)^2 + (0.68-0.516)^2 + (0.65-0.516)^2 + (0.22-0.516)^2]$
- $= \frac{1}{5}[0.0416 + 0.0424 + 0.0269 + 0.0180 + 0.0876] = \mathbf{0.0433}$

**Application fairness** — Disagreement attendu par sous-groupe :

| Sous-groupe | Disagreement attendu | Lien avec les résultats |
|---|---|---|
| Female × nurse | ~0.05 (faible) | Profil bien appris, cohérent avec ∆EO bas pour ce sous-groupe |
| Male × nurse | ~0.35 (élevé) | Profil ambigu, cohérent avec ∆EO=35% |
| Male × surgeon | ~0.08 (faible) | Profil bien appris |
| Female × surgeon | ~0.30 (élevé) | Profil ambigu, cohérent avec ∆EO=44.5% |

> ⚠️ Ces valeurs de disagreement sont des **estimations qualitatives cohérentes** avec les ∆EO observés, pas des résultats expérimentaux. L'expérience réelle (entraîner 5 FairMLPs) donnerait les vrais chiffres.

---

### 2.4 ECE — Expected Calibration Error

**Formulation** :

$$\text{ECE} = \sum_{i=1}^{m} \frac{|B_i|}{N} \cdot |\text{accuracy}(B_i) - \text{confidence}(B_i)|$$

- $B_i$ : ensemble des prédictions dont la confiance max tombe dans le bin $i$
- $\text{confidence}(B_i) = \frac{1}{|B_i|} \sum_{x \in B_i} \max_k P(Y=k|x)$
- $\text{accuracy}(B_i) = \frac{1}{|B_i|} \sum_{x \in B_i} \mathbb{1}[\hat{y}(x) = y(x)]$
- ECE = 0 → calibration parfaite

**Exemple numérique — 12 prédictions, m=3 bins** :

| # | Conf. max | Correct ? | Genre | Bin |
|---|---|---|---|---|
| 1 | 0.62 | ❌ | M | [0.5, 0.7) |
| 2 | 0.68 | ✅ | F | [0.5, 0.7) |
| 3 | 0.55 | ❌ | M | [0.5, 0.7) |
| 4 | 0.65 | ✅ | F | [0.5, 0.7) |
| 5 | 0.78 | ✅ | F | [0.7, 0.9) |
| 6 | 0.82 | ✅ | M | [0.7, 0.9) |
| 7 | 0.75 | ❌ | M | [0.7, 0.9) |
| 8 | 0.85 | ✅ | F | [0.7, 0.9) |
| 9 | 0.93 | ✅ | F | [0.9, 1.0] |
| 10 | 0.97 | ✅ | M | [0.9, 1.0] |
| 11 | 0.91 | ✅ | F | [0.9, 1.0] |
| 12 | 0.95 | ❌ | M | [0.9, 1.0] |

**Calcul ECE global** :

| Bin | $|B_i|$ | Conf. moy | Acc | $|$Acc − Conf$|$ | Contribution |
|---|---|---|---|---|---|
| [0.5, 0.7) | 4 | 0.625 | 2/4 = 0.500 | 0.125 | $\frac{4}{12} \times 0.125 = 0.042$ |
| [0.7, 0.9) | 4 | 0.800 | 3/4 = 0.750 | 0.050 | $\frac{4}{12} \times 0.050 = 0.017$ |
| [0.9, 1.0] | 4 | 0.940 | 3/4 = 0.750 | 0.190 | $\frac{4}{12} \times 0.190 = 0.063$ |

$$\text{ECE}_{\text{global}} = 0.042 + 0.017 + 0.063 = \mathbf{0.122}$$

**Calcul ECE par genre** :

**Femmes** (échantillons 2, 4, 5, 8, 9, 11) — toutes correctes (6/6) :

| Bin | $|B_i|$ | Conf. moy | Acc | Contribution |
|---|---|---|---|---|
| [0.5, 0.7) | 2 | 0.665 | 1.0 | $\frac{2}{6} \times 0.335 = 0.112$ |
| [0.7, 0.9) | 2 | 0.815 | 1.0 | $\frac{2}{6} \times 0.185 = 0.062$ |
| [0.9, 1.0] | 2 | 0.920 | 1.0 | $\frac{2}{6} \times 0.080 = 0.027$ |

$$\text{ECE}_F = 0.112 + 0.062 + 0.027 = \mathbf{0.201}$$

Ici l'ECE est élevé **non pas parce que le modèle se trompe**, mais parce qu'il **sous-estime sa propre fiabilité** sur les femmes (confiance 66% alors qu'accuracy = 100%).

**Hommes** (échantillons 1, 3, 6, 7, 10, 12) — 2 correctes sur 6 :

| Bin | $|B_i|$ | Conf. moy | Acc | Contribution |
|---|---|---|---|---|
| [0.5, 0.7) | 2 | 0.585 | 0/2 = 0.0 | $\frac{2}{6} \times 0.585 = 0.195$ |
| [0.7, 0.9) | 2 | 0.785 | 1/2 = 0.5 | $\frac{2}{6} \times 0.285 = 0.095$ |
| [0.9, 1.0] | 2 | 0.960 | 1/2 = 0.5 | $\frac{2}{6} \times 0.460 = 0.153$ |

$$\text{ECE}_M = 0.195 + 0.095 + 0.153 = \mathbf{0.443}$$

**Nouveau diagnostic de biais** :

$$\Delta\text{ECE} = |\text{ECE}_F - \text{ECE}_M| = |0.201 - 0.443| = \mathbf{0.242}$$

**Interprétation** : Le modèle est **massivement mieux calibré pour les femmes que pour les hommes** dans cet exemple. L'ECE_M de 0.443 signifie que le modèle est **overconfident** sur les hommes — il affiche des confiances élevées (0.96) alors qu'il ne prédit correctement qu'une fois sur deux.

Ce ∆ECE est un diagnostic que ∆DP et ∆EO ne capturent **pas**. Un modèle peut avoir ∆EO=0 (même taux de vrais positifs par genre) mais un ∆ECE élevé (confiance trompeuse pour un genre).

> **Lien avec le cours** : "Quantifier l'incertitude prévient les prédictions overconfident qui peuvent mener à des décisions biaisées ou injustes." Le ∆ECE est la matérialisation exacte de ce principe appliqué à la fairness.

> ⚠️ Ces 12 exemples sont illustratifs. Les vrais chiffres viendraient du test set complet (1000 exemples).

---

### 2.5 Approche Bayésienne vs MAP — Ce que fait mon FairMLP

**Mon FairMLP actuel = optimisation MAP** :

$$\omega^* = \arg\max_\omega [\log P(D_l|\omega) + \log P(\omega)]$$

- $\log P(D_l|\omega)$ = cross-entropy loss
- $\log P(\omega)$ = prior gaussien → `weight_decay=1e-4` dans Adam, soit $P(\omega) = \mathcal{N}(0, \sigma^2 I)$

Le résultat est un **unique jeu de poids** $\omega^*$ — aucune incertitude sur les paramètres.

**L'approche BNN (marginalisation)** :

$$P(Y|X) = \int P(Y|X, \omega) \, P(\omega|D) \, d\omega \approx \frac{1}{M}\sum_{i=1}^{M} P(Y|X, \omega_i), \quad \omega_i \sim P(\omega|D)$$

En pratique, on approxime $P(\omega|D)$ par :
- **MC Dropout** : $\omega_i = \omega \odot b_i$ (approximation variationnelle)
- **Deep Ensembles** : chaque $\omega_j$ est un mode différent du paysage de perte
- **ABNN/BNL** : perturbation gaussienne des couches de normalisation

**Variational Inference** — approximer $P(\omega|D)$ par $q_\lambda(\omega)$ en maximisant l'ELBO :

$$\text{ELBO}(\lambda) = \mathbb{E}_q[\log P(D_l|\omega)] - \text{KL}(q_\lambda(\omega|D_l) \| P(\omega))$$

Le premier terme est la vraisemblance attendue (mon CE loss actuel), le second est la régularisation KL (mon weight_decay est une version simplifiée).

**Conclusion** : Mon FairMLP fait déjà du "proto-bayésien" via weight decay + dropout, mais sans en extraire l'incertitude. MC Dropout est la façon la plus simple de passer de MAP à approximation bayésienne.

---

### 2.6 ABNN — Conversion post-hoc DNN→BNN

**Formulation** :

Remplacer les couches de normalisation par des BNL :

$$\text{BNL}(h_j) = \frac{h_j - \hat{\mu}_j}{\hat{\sigma}_j} \times \gamma_j(1 + \epsilon_j) + \beta_j, \quad \epsilon_j \sim \mathcal{N}(0, I)$$

À l'inférence, pour $L$ tirages de $\epsilon$ et $M$ configurations de poids :

$$P(y|x, D) \approx \frac{1}{ML}\sum_{l=1}^{L}\sum_{m=1}^{M} P(y|x, \omega_m, \epsilon_l)$$

**Application à mon FairMLP** :

Mon MLP actuel : `Linear(768,128) → ReLU → Dropout → Linear(128,64) → ReLU → Dropout → Linear(64,5)`

Version avec BNL :
1. Ajouter `LayerNorm` après chaque ReLU
2. Entraîner normalement
3. Remplacer chaque LayerNorm par BNL (ajouter $\epsilon \sim \mathcal{N}(0,1)$)
4. Fine-tuner γ et β pendant 2–3 epochs
5. À l'inférence : $L$ tirages de $\epsilon$ → distribution prédictive

Comparé à MC Dropout, l'ABNN offre une **incertitude bayésienne formelle** plutôt qu'une approximation variationnelle. Mais MC Dropout est plus simple à implémenter et suffisant pour mon échelle de modèle.

---

## 3. CLASSIFICATION — Taxonomie

```
MON TRAVAIL ACCOMPLI
│
├── 🔵 A. ACCOMPLI — HORS SCOPE UNCERTAINTY
│   │   (Techniques fairness pures, sans lien avec l'UQ)
│   │
│   ├── Fairness Loss combinée CE + λ·penalty (∆DP ou ∆EO)
│   ├── Sweep λ ∈ {0, 0.01, 0.1, 0.5, 1.0} × {dp, eo}
│   ├── SMOTE sur embeddings avec label combiné genre×profession
│   ├── Undersampling des groupes majoritaires
│   ├── Class weighting (balanced, balanced_subsample)
│   ├── Sample weighting genre×profession
│   ├── Augmentation par bruit gaussien sur embeddings
│   ├── Post-processing : seuils différenciés par genre
│   ├── Métriques ∆DP, ∆EO, ∆EqOdds, Predictive Parity
│   ├── Feature importance Gini + corrélation point-bisériale
│   ├── Analyse intersectionnelle genre × profession
│   ├── Tests statistiques Chi², KS, Mann-Whitney
│   └── Comparaison 5 modèles d'embedding
│
├── 🟢 B. ACCOMPLI — ALIGNÉ AVEC LA THÉORIE UQ
│   │   (Ce que j'ai fait et qui s'inscrit dans les concepts du cours)
│   │
│   ├── Random Forest = Bagging Ensemble
│   │   → 100 arbres sur bootstrap samples, vote majoritaire
│   │   → Concept : méthodes d'ensemble pour robustesse
│   │
│   ├── Dropout (p=0.3) dans FairMLP
│   │   → Régularisation + ensemble implicite de sous-réseaux
│   │   → Concept : MC Dropout (utilisé en régularisation, pas en UQ)
│   │
│   ├── Bootstrap CI (2000 iter) sur métriques fairness
│   │   → CI ∆EO : [0.112, 0.251] — biais robuste
│   │   → Concept : incertitude épistémique (paramétrique)
│   │
│   ├── Courbes de calibration par genre
│   │   → Distribution max_proba, correct vs incorrect
│   │   → Concept : overconfidence et calibration (ECE qualitatif)
│   │
│   ├── Stabilité temporelle 2017 vs 2018
│   │   → acc 86.6% → 91.6%, ∆EO 14.7% → 18.9%
│   │   → Concept : distribution shift (sensoriel + sémantique)
│   │
│   ├── Sensibilité taille échantillon N=500→8000
│   │   → Gaps augmentent sous N=2000
│   │   → Concept : incertitude épistémique réductible
│   │
│   ├── 5-Fold Cross-validation
│   │   → std(∆EO) = 0.027, 3× plus que std(acc)
│   │   → Concept : variance épistémique du biais
│   │
│   ├── Data Imbalance genre×profession
│   │   → Source fondamentale du biais
│   │   → Concept : source d'incertitude dans les données
│   │
│   └── Weight decay (1e-4) dans Adam
│       → Prior gaussien sur les poids = proto-bayésien
│       → Concept : régularisation MAP (slide ELBO)
│
├── 🟡 C. APPLICABLE IMMÉDIATEMENT — EFFORT MODÉRÉ
│   │   (Techniques du cours implémentables sur mon pipeline actuel)
│   │
│   ├── C1. MC Dropout sur FairMLP ⭐ PRIORITÉ HAUTE
│   │   ├── Quoi : model.train() à l'inférence, T=30 passes
│   │   ├── Mesurer : H_total, MI par genre×profession
│   │   ├── Nouveau : ∆MI = biais d'incertitude épistémique
│   │   └── Effort : ~4h
│   │
│   ├── C2. Entropie prédictive comme diagnostic ⭐ PRIORITÉ HAUTE
│   │   ├── Quoi : H(P(Y|x)) par prédiction, moyenné par genre
│   │   ├── Nouveau : ∆H par profession = biais de confiance
│   │   └── Effort : ~2h
│   │
│   ├── C3. ECE global + ECE par genre + ∆ECE
│   │   ├── Quoi : score de calibration quantitatif
│   │   ├── Nouveau : ∆ECE = biais de calibration
│   │   └── Effort : ~3h
│   │
│   ├── C4. Deep Ensemble de 5 FairMLPs
│   │   ├── Quoi : même config λ=0.1 eo, 5 seeds
│   │   ├── Mesurer : disagreement, variance par genre
│   │   └── Effort : ~6h
│   │
│   ├── C5. Rejection basée sur incertitude
│   │   ├── Quoi : rejeter X% des prédictions les plus incertaines
│   │   ├── Mesurer : ∆EO en fonction du taux de rejet
│   │   └── Effort : ~3h
│   │
│   └── C6. Formaliser MCP (Max Class Probability)
│       ├── Quoi : renommer l'analyse NB07 avec vocabulaire UQ
│       └── Effort : ~1h
│
└── 🔴 D. EXTENSIONS AVANCÉES — EFFORT CONSÉQUENT
    │   (Pistes de recherche nécessitant un travail supplémentaire)
    │
    ├── D1. ABNN — Conversion FairMLP → BNN post-hoc
    │   └── Ajouter LayerNorm → BNL, fine-tuner → ~2 semaines
    │
    ├── D2. MIMO — Multi-Input Multi-Output FairMLP
    │   └── Un modèle, 3 sous-réseaux, 3 critères fairness → ~2 sem.
    │
    ├── D3. BatchEnsemble — Poids partagés + rank-1
    │   └── W_j = W · (r_j s_j^T) → ~3 semaines
    │
    ├── D4. Packed-Ensembles — Scale-up transformers
    │   └── Grouped convolutions, pertinent si fine-tuning → ~1 mois
    │
    ├── D5. Variational Inference complète (BNN from scratch)
    │   └── ELBO, reparameterization trick → ~1–2 mois
    │
    └── D6. TRADI — Tracking poids pendant entraînement
        └── Kalman filtering sur la posterior → ~1 mois
```

```
THÉORIE UQ APPLICABLE AU PROJET
│
├── 🎯 IMPACT DIRECT SUR FAIRNESS
│   ├── MC Dropout → ∆MI par genre (biais épistémique)
│   ├── Entropie → ∆H par genre (biais de confiance)
│   ├── ECE par genre → ∆ECE (biais de calibration)
│   └── Rejection → ∆EO post-rejection
│
├── 📊 AMÉLIORATION ROBUSTESSE
│   ├── Deep Ensembles → variance des métriques fairness
│   ├── MCP formalisé → vocabulaire standardisé UQ
│   └── Sparsification → correspondance incertitude ↔ erreur
│
└── 🔬 NOUVELLES DIRECTIONS DE RECHERCHE
    ├── ABNN → incertitude bayésienne formelle sur fairness
    ├── MIMO → multi-critères fairness en un modèle
    ├── Uncertainty-Aware Fairness Loss (nouveau)
    └── OOD Detection → sous-populations mal servies
```

---

## 4. ROADMAP — 3 Niveaux d'implémentation

### 🟢 NIVEAU 1 — Diagnostics UQ (1–2 semaines, ~19h)

**Objectif** : Enrichir le pipeline avec des métriques d'incertitude sans changer l'architecture.

| # | Technique | Cible | Livrable | Heures |
|---|---|---|---|---|
| 1.1 | Entropie H par genre×profession | NB08 + metrics.py | ∆H par profession, graphiques | 2h |
| 1.2 | ECE global + ECE_F + ECE_M + ∆ECE | NB07 + metrics.py | Score ECE, reliability diagrams annotés | 3h |
| 1.3 | MC Dropout (T=30) | NB10 | MI par genre×profession, histogrammes | 4h |
| 1.4 | MCP formalisé | NB07 | Renommage + interprétation UQ | 1h |
| 1.5 | Rejection curve | NB08 ou nouveau | ∆EO vs taux de rejet (5%, 10%, 20%) | 3h |
| 1.6 | NB15 "Uncertainty-Aware Fairness Audit" | Nouveau notebook | Synthèse UQ × Fairness | 6h |

**Code prêt à l'emploi** :

```python
# === MC Dropout (à ajouter dans NB10) ===
def mc_dropout_predict(model, X_tensor, T=30):
    """MC Dropout : T forward passes avec dropout activé."""
    model.train()  # ← CLEF : garder dropout actif
    all_probs = []
    with torch.no_grad():
        for _ in range(T):
            logits = model(X_tensor)
            probs = torch.softmax(logits, dim=1)
            all_probs.append(probs.cpu().numpy())
    all_probs = np.stack(all_probs)              # (T, N, K)
    mean_probs = all_probs.mean(axis=0)           # (N, K)
    H_total = -np.sum(mean_probs * np.log(mean_probs + 1e-10), axis=1)
    H_per_pass = -np.sum(all_probs * np.log(all_probs + 1e-10), axis=2)
    H_aleatoric = H_per_pass.mean(axis=0)
    MI = H_total - H_aleatoric                    # Information mutuelle
    return mean_probs, H_total, H_aleatoric, MI

# === ECE par genre (à ajouter dans utils/fairness_metrics.py) ===
def compute_ece(y_true, y_proba, n_bins=10):
    """Expected Calibration Error."""
    confidences = y_proba.max(axis=1)
    predictions = y_proba.argmax(axis=1)
    accuracies = (predictions == y_true).astype(float)
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if mask.sum() > 0:
            ece += (mask.sum() / len(y_true)) * abs(accuracies[mask].mean() - confidences[mask].mean())
    return ece

def compute_ece_by_gender(y_true, y_proba, genders, n_bins=10):
    """ECE par genre + ∆ECE."""
    results = {}
    for g in np.unique(genders):
        mask = genders == g
        results[f'ECE_{g}'] = compute_ece(y_true[mask], y_proba[mask], n_bins)
    groups = list(np.unique(genders))
    results['delta_ECE'] = abs(results[f'ECE_{groups[0]}'] - results[f'ECE_{groups[1]}'])
    return results

# === Entropie par genre (à ajouter dans utils/fairness_metrics.py) ===
def compute_entropy_fairness(y_proba, genders):
    """Entropie prédictive moyenne par genre + ∆H."""
    H = -np.sum(y_proba * np.log(y_proba + 1e-10), axis=1)
    results = {}
    for g in np.unique(genders):
        results[f'H_mean_{g}'] = H[genders == g].mean()
    groups = list(np.unique(genders))
    results['delta_H'] = abs(results[f'H_mean_{groups[0]}'] - results[f'H_mean_{groups[1]}'])
    return results, H
```

---

### 🟡 NIVEAU 2 — UQ active pour fairness (1 mois, ~32h)

**Objectif** : Utiliser l'incertitude pour **améliorer activement** la fairness.

| # | Technique | Description | Heures |
|---|---|---|---|
| 2.1 | Deep Ensemble 5 FairMLPs | Même λ=0.1 eo, 5 seeds → disagreement par genre | 6h |
| 2.2 | **Uncertainty-Aware Fairness Loss** | 3e terme dans la loss (voir ci-dessous) | 10h |
| 2.3 | Rejection curve complète | ∆EO en fonction du taux de rejet (0%→30%) | 4h |
| 2.4 | Temperature Scaling par genre | Calibration post-hoc avec $T_F \neq T_M$ | 4h |
| 2.5 | NLL par genre | Negative Log-Likelihood comme mesure de qualité probabiliste | 2h |
| 2.6 | Benchmark consolidé UQ×Fairness | Tableau : accuracy, ∆EO, ECE, ∆ECE, MI, ∆MI | 6h |

**Idée clé — Uncertainty-Aware Fairness Loss** :

$$\mathcal{L} = \underbrace{\text{CE}(f_\omega(x), y)}_{\text{classification}} + \lambda_f \cdot \underbrace{|\text{TPR}_F - \text{TPR}_M|}_{\text{equality of opportunity}} + \lambda_u \cdot \underbrace{|\mathbb{E}_F[H] - \mathbb{E}_M[H]|}_{\text{uncertainty equity}}$$

Le 3e terme force le modèle à avoir le **même niveau de confiance** pour les deux genres. Un modèle qui prédit "nurse" avec 95% de confiance pour les femmes mais 60% pour les hommes est **injuste dans sa certitude**, même si les deux prédictions sont correctes. C'est le lien direct entre le cours ("éviter la surconfiance pour prévenir les décisions injustes") et le projet.

---

### 🔴 NIVEAU 3 — Recherche avancée (3+ mois)

**Objectif** : Contributions originales à l'intersection UQ × Fairness.

| # | Direction | Contribution potentielle |
|---|---|---|
| 3.1 | **ABNN pour Fairness** | Convertir FairMLP→BNN via BNL, comparer MI bayésienne vs MI MC Dropout. Première application d'ABNN à un problème de biais NLP. |
| 3.2 | **Multi-Criteria Fair Ensemble** | Deep Ensemble où chaque membre optimise un critère fairness différent (DP, EO, EqOdds). L'ensemble satisfait tous les critères simultanément. |
| 3.3 | **OOD Fairness** | Traiter Female×surgeon comme "OOD interne" → appliquer AUROC/FPR95 pour détecter les sous-populations mal servies. Pont entre OOD detection et audit de biais. |
| 3.4 | **Conformal Prediction + Fairness** | Prediction sets avec couverture $1-\alpha$ garantie par genre → "equalized coverage". Au-delà du PDF mais prolongement naturel. |
| 3.5 | **TRADI pour Fairness Monitoring** | Tracker la distribution des poids pendant l'entraînement pour détecter **quand** le biais s'installe dans les poids du réseau. |

---

## 5. RÉFÉRENCES ACADÉMIQUES

### 5A. Références du cours directement applicables

| Ref | Article | Année | Application projet |
|---|---|---|---|
| [12] | Gal & Ghahramani, "Dropout as a Bayesian Approximation" | ICML 2016 | **MC Dropout** — Niveau 1 |
| [13] | Lakshminarayanan et al., "Simple and Scalable Predictive UQ using Deep Ensembles" | NeurIPS 2017 | **Deep Ensembles** — Niveau 2 |
| [9] | Guo et al., "On Calibration of Modern Neural Networks" | ICML 2017 | **ECE + Temperature Scaling** — Niveaux 1–2 |
| [25] | Laurent, Franchi et al., "Packed-Ensembles for Efficient UQ" | ICLR 2023 | **Packed-Ensembles** — Niveau 3 |
| [17] | Blundell et al., "Weight Uncertainty in Neural Networks" | 2015 | **BNN fondement** |
| [24] | Franchi et al., "TRADI: Tracking DNN Weight Distributions" | ECCV 2020 | **TRADI** — Niveau 3 |
| [22] | Wen et al., "BatchEnsemble" | 2020 | **BatchEnsemble** — Niveau 3 |
| [11] | Havasi et al., "Training Independent Subnetworks (MIMO)" | 2020 | **MIMO** — Niveau 3 |
| [8] | Hüllermeier & Waegeman, "Aleatoric and Epistemic Uncertainty in ML" | ML Journal 2021 | **Cadre théorique** |
| [7] | Gawlikowski et al., "A Survey of Uncertainty in DNNs" | AI Review 2023 | **Survey** |
| [14] | Kendall & Gal, "What Uncertainties Do We Need in Bayesian DL?" | NeurIPS 2017 | **Décomposition** aléatoire/épistémique |

### 5B. Articles reliant Uncertainty × Fairness (hors cours)

| Article | Année | Venue | Apport | Lien avec le projet |
|---|---|---|---|---|
| Pleiss et al., "On Fairness and Calibration" | 2017 | NeurIPS | Un modèle calibré par groupe satisfait automatiquement la sufficiency (une notion de fairness). | Fondement théorique de l'approche ∆ECE. |
| Barda et al., "Addressing Bias by Improving Subpopulation Calibration" | 2021 | JAMIA | La miscalibration est pire pour les sous-groupes minoritaires en clinique. | Directement applicable au contexte médical. |
| Kompa et al., "Second Opinion Needed: Communicating Uncertainty in Medical ML" | 2021 | npj Digital Medicine | En médecine, l'incertitude doit être communiquée. MC Dropout recommandé. | Le contexte médical rend l'UQ **nécessaire**. |
| Chuang & Mroueh, "Fair Mixup: Fairness via Interpolation" | 2021 | ICLR | Régularisation par interpolation améliore robustesse + fairness. | Complémentaire à l'augmentation bruit (NB09). |
| Zhao et al., "Calibration in Deep Learning: A Survey" | 2024 | ACM Computing Surveys | Calibration et fairness sont liées. | Justification théorique de la roadmap. |

### 5C. Repositories et outils

| Outil | URL | Usage |
|---|---|---|
| **TorchUncertainty** | https://github.com/ENSTA-U2IS/torch-uncertainty | Labo de Franchi — MC Dropout, Deep Ensembles, Packed-Ensembles, MIMO, ECE, MI |
| **Awesome Uncertainty DL** | https://github.com/ENSTA-U2IS/awesome-uncertainty-deeplearning | Liste curatée de papers UQ |
| **Uncertainty Toolbox** | https://github.com/uncertainty-toolbox/uncertainty-toolbox | Métriques UQ (calibration, sharpness) |
| **Netcal** | https://github.com/fabiankueppers/calibration-framework | Framework calibration (ECE, MCE) |
| **AIF360** (IBM) | https://github.com/Trusted-AI/AIF360 | Fairness toolkit |
| **Fairlearn** (Microsoft) | https://github.com/fairlearn/fairlearn | Métriques et algorithmes fairness |

---

## Résumé exécutif

| Dimension | État actuel | + Niveau 1 (UQ diagnostique) | + Niveau 2 (UQ active) |
|---|---|---|---|
| **Accuracy** | 97.5% | 97.5% (inchangé) | ~97% (variance ensemble) |
| **∆EO** | 2.5% | 2.5% (inchangé) | Potentiellement <2% (rejection) |
| **∆DP** | 16.1% | 16.1% (structurel) | ~16% (structurel dataset) |
| **Calibration** | Qualitative | ECE + ∆ECE quantifiés | Temperature scaling par genre |
| **Incertitude** | Non mesurée | H, MI par genre (MC Dropout) | Deep Ensemble + MI |
| **Types de biais détectés** | ∆DP, ∆EO, ∆EqOdds, ∆PPV | + ∆H (confiance), ∆ECE (calibration), ∆MI (épistémique) | + Disagreement, rejection curve |
| **Robustesse** | Bootstrap CI, 5-Fold CV | + variance MC Dropout | + variance inter-modèles |
| **Valeur académique** | Projet complet | Audit UQ×Fairness | Contribution originale |
