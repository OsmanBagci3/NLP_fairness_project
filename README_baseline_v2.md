# Fairness Analysis — Medical Bios Classification
### `fairness_baseline_v2.ipynb`

Notebook d'analyse de la fairness d'un classifieur de professions médicales basé sur **DistilBERT + Régression Logistique**, avec comparaison de deux stratégies de débiaisage lexical (D1 et D1n).

---

## Objectif

Classifier des biographies médicales en 5 professions, puis **quantifier et réduire les biais de prédiction liés au genre** (Male / Female) en comparant différentes versions du texte d'entrée.

---

## Dataset

**[coastalcph/medical-bios](https://huggingface.co/datasets/coastalcph/medical-bios)** — biographies de professionnels de santé.

| Split | Exemples |
|-------|----------|
| Train | 8 000 |
| Validation | 1 000 |
| Test | 1 000 |

**5 professions cibles** : `nurse`, `surgeon`, `psychiatrist`, `physician`, `dentist` *(selon les classes du dataset)*.

---

## Pipeline

```
full_text (D0)
    │
    ├─ D1  : masquage des mots-métiers → [PROF]
    │         (regex sur ~40+ termes : physician, surgeon, nurse, ...)
    │
    └─ D1n : D1 + neutralisation des marqueurs de genre
              (he/she → [PRONOUN], Mr/Mrs → [TITLE],
               mother/father → [PARENT], wife/husband → [SPOUSE], ...)

Pour D1 et D1n :
    DistilBERT (frozen, token [CLS])
        → embeddings 768 dims
        → Régression Logistique (C=1.0, max_iter=1000)
        → Évaluation fairness par genre
```

---

## Modèle

| Composant | Détail |
|-----------|--------|
| Encodeur | `distilbert-base-uncased` (frozen) |
| Embedding | Token `[CLS]` du dernier hidden state (768 dims) |
| Classifieur | `LogisticRegression` (C=1.0, max_iter=1000, random_state=42) |
| Reproductibilité | `SEED = 42` |
| Cache embeddings | Fichiers `.npy` dans `fairness_nlp_cache/embeddings_baseline/` |

---

## Traitements des données

| Version | Description |
|---------|-------------|
| **D1** | Mots-métiers remplacés par `[PROF]` — supprime le *label leakage* lexical |
| **D1n** | D1 + neutralisation des indices de genre explicites (pronoms, titres, relations familiales) |

### Exemples de substitutions D1n

| Signal de genre | Remplacement |
|----------------|-------------|
| `he`, `she`, `him`, `himself`, `herself` | `[PRONOUN]` |
| `his`, `her` | `[POSS]` |
| `Mr`, `Mrs`, `Ms`, `Miss`, `Dr` | `[TITLE]` |
| `mother`, `father` | `[PARENT]` |
| `wife`, `husband` | `[SPOUSE]` |
| `son`, `daughter` | `[CHILD]` |
| `brother`, `sister` | `[SIBLING]` |
| `boyfriend`, `girlfriend` | `[PARTNER]` |

---

## Structure du notebook

| Section | Contenu |
|---------|---------|
| **Section 1** | Import des librairies & configuration (device, seed) |
| **Section 2** | Téléchargement et chargement du dataset Medical Bios |
| **Section 3** | Encodage des labels (LabelEncoder fit sur train uniquement) + création D1/D1n |
| **Section 4** | Extraction des embeddings DistilBERT (D1 et D1n) + entraînement LR |
| **Section 5** | Résultats globaux (Accuracy, F1 macro, Recall macro) |
| **Section 6** | F1-score par profession (classification report + figure) |
| **Section 7** | Analyse de fairness par genre : F1/Recall par profession, gender gap directionnel, matrices de confusion |
| **Section 8** | Comparaison D1 vs D1n : tableau et figures comparatives |

---

## Métriques de fairness

- **Equal Opportunity gap** : écart de Recall macro entre Hommes et Femmes  
  *(métrique principale — Hardt et al., 2016)*
- **Gender gap F1** : écart de F1 macro entre Hommes et Femmes
- Les gaps sont exprimés de façon directionnelle : `+ = avantage Hommes`, `− = avantage Femmes`
- Seuil d'alerte : `|gap| > 0.05` (affiché en rouge dans les figures)

---

## Figures générées

### Section 6 — F1 par profession

**`results_by_profession.png`**

> F1-score par profession sur le set de validation, avec la ligne F1 macro en référence.

---

### Section 7 — Analyse fairness par genre (D1)

**`results_fairness_by_gender.png`**

> Figure 2×2 :
> - *Ligne 1* : F1-score par profession × genre (barres côte à côte) + gap directionnel (H−F)
> - *Ligne 2* : Recall (Equal Opportunity) par profession × genre + gap directionnel (H−F)
>
> Barres rouges = |gap| > 0.05 (biais significatif), vertes = biais acceptable.

**`confusion_matrix_by_gender.png`**

> Matrices de confusion normalisées (par vraie classe) pour les Hommes et les Femmes séparément.  
> Permet d'identifier les professions systématiquement mal classées selon le genre.

---

### Section 8 — Comparaison D1 vs D1n

**`compare_d1_d1n_performance.png`**

> Performances globales comparées : F1 macro et Recall macro pour D1 et D1n.

**`compare_d1_d1n_gap.png`**

> Gender gap comparé D1 vs D1n :
> - Gap Recall (Equal Opportunity)
> - Gap F1
>
> Permet de vérifier si la neutralisation du genre (D1n) réduit ou creuse le biais.

**`compare_d1_d1n_recall_by_profession.png`**

> Recall par profession et par genre, pour D1 et D1n côte à côte (Hommes / Femmes séparément).

---

## Dépendances

```
transformers
torch
scikit-learn
datasets
numpy
pandas
matplotlib
```

Installation :
```bash
pip install transformers scikit-learn datasets torch numpy pandas matplotlib
```

---

## Exécution

Le notebook est conçu pour **Google Colab** (GPU recommandé). En local, remplacer la cellule de montage Google Drive par un chemin local :

```python
import os
DRIVE_DIR = os.path.join(os.getcwd(), "fairness_nlp_cache")
os.makedirs(DRIVE_DIR, exist_ok=True)
```

Les embeddings sont mis en cache dans `fairness_nlp_cache/embeddings_baseline/` pour éviter de les recalculer à chaque session.

---

## Résultats

### Performances globales (validation)

| Version | F1 macro | Recall macro | EOpp gap (H−F) | F1 gap (H−F) |
|---------|----------|-------------|----------------|--------------|
| **D1**  | **87.84%** | **87.48%** | ⚠️ +8.37% | ⚠️ +6.56% |
| **D1n** | **86.77%** | **86.55%** | ✅ +4.79% | ✅ +2.99% |

> La neutralisation du genre (D1n) réduit l'EOpp gap de **−3.58 points** et le F1 gap de **−3.57 points**, au prix d'une légère baisse de performance globale (−1.07 pts F1 macro).

---

### F1-score par profession — D1

| Profession | F1-score |
|------------|----------|
| dentist | 0.936 |
| psychologist | 0.923 |
| surgeon | 0.862 |
| nurse | 0.858 |
| physician | 0.812 |
| **Macro** | **0.878** |

---

### Analyse fairness par genre — D1

#### F1-score par profession × genre

| Profession | Hommes | Femmes | Gap (H−F) | Biais |
|------------|--------|--------|-----------|-------|
| dentist | 0.94 | 0.93 | +0.009 | ✅ |
| nurse | 0.75 | 0.86 | −0.115 | ⚠️ |
| physician | 0.86 | 0.77 | +0.090 | ⚠️ |
| psychologist | 0.93 | 0.92 | +0.005 | ✅ |
| surgeon | 0.90 | 0.56 | **+0.338** | 🔴 |

#### Recall (Equal Opportunity) par profession × genre

| Profession | Hommes | Femmes | Gap (H−F) | Biais |
|------------|--------|--------|-----------|-------|
| dentist | 0.92 | 0.97 | −0.050 | ✅ |
| nurse | 0.75 | 0.89 | −0.142 | ⚠️ |
| physician | 0.86 | 0.72 | +0.137 | ⚠️ |
| psychologist | 0.96 | 0.93 | +0.027 | ✅ |
| surgeon | 0.90 | 0.45 | **+0.447** | 🔴 |

> **Constat majeur** : la profession `surgeon` présente un biais extrême — le modèle reconnaît 90% des chirurgiens hommes mais seulement 45% des chirurgiens femmes (recall). Ce biais est fortement corrélé à la présence de pronoms genrés dans les bios.

---

### Matrices de confusion par genre — D1

#### Hommes
| Vraie classe \ Prédite | dentist | nurse | physician | psychologist | surgeon |
|------------------------|---------|-------|-----------|--------------|---------|
| dentist | **0.92** | 0.00 | 0.02 | 0.03 | 0.04 |
| nurse | 0.08 | **0.75** | 0.00 | 0.00 | 0.17 |
| physician | 0.00 | 0.01 | **0.86** | 0.07 | 0.07 |
| psychologist | 0.02 | 0.00 | 0.02 | **0.96** | 0.01 |
| surgeon | 0.01 | 0.02 | 0.05 | 0.03 | **0.90** |

#### Femmes
| Vraie classe \ Prédite | dentist | nurse | physician | psychologist | surgeon |
|------------------------|---------|-------|-----------|--------------|---------|
| dentist | **0.97** | 0.03 | 0.00 | 0.00 | 0.00 |
| nurse | 0.00 | **0.89** | 0.05 | 0.04 | 0.01 |
| physician | 0.02 | 0.19 | **0.72** | 0.07 | 0.00 |
| psychologist | 0.00 | 0.06 | 0.01 | **0.93** | 0.00 |
| surgeon | 0.20 | 0.20 | 0.10 | 0.05 | **0.45** |

> Pour les femmes chirurgiennes, 55% sont mal classées (confondues avec dentist 20%, nurse 20%, physician 10%). Pour les hommes, 90% sont correctement identifiés.

---

### Comparaison D1 vs D1n — Gender gap

| Métrique | D1 | D1n | Δ (D1n−D1) |
|----------|-----|------|------------|
| F1 macro | 87.84% | 86.77% | −1.07 pts |
| Recall macro | 87.48% | 86.55% | −0.93 pts |
| **EOpp gap (H−F)** | ⚠️ **+8.37%** | ✅ **+4.79%** | **−3.58 pts** |
| **F1 gap (H−F)** | ⚠️ **+6.56%** | ✅ **+2.99%** | **−3.57 pts** |

### Recall D1 vs D1n par profession × genre

#### Hommes
| Profession | D1 | D1n |
|------------|-----|------|
| dentist | 0.92 | 0.88 |
| nurse | 0.75 | 0.83 |
| physician | 0.86 | 0.81 |
| psychologist | 0.95 | 0.94 |
| surgeon | 0.90 | 0.87 |

#### Femmes
| Profession | D1 | D1n |
|------------|-----|------|
| dentist | 0.97 | 0.95 |
| nurse | 0.89 | 0.88 |
| physician | 0.69 | 0.72 |
| psychologist | 0.93 | 0.92 |
| surgeon | 0.45 | 0.60 |

> **D1n améliore fortement le recall surgeon pour les femmes** (+15 pts : 0.45 → 0.60), ce qui explique la réduction du gender gap global.
