# 📊 Benchmark Groupe - État des Lieux

> **Objectif** : Faire le point sur nos projets actuels
> **Date** : Avril 2026  
> **Participants** : Antoine D, Antoine L, Osman, Leo, Saad

---

## 🎯 **VUE D'ENSEMBLE**

| **Critère** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|-------------|---------------|---------------|-----------|---------|-----------|
| **👤 Projet** | NLP Fairness Médical | [Nom projet] | [Nom projet] | NLP Fairness Médical | [Nom projet] |
| **📊 Dataset** | Medical Bios (8k) | [Dataset + taille] | [Dataset + taille] | Medical Bios (8k train / 1k val / 1k test) | [Dataset + taille] |
| **🎯 Tâche** | Classification 5 professions | [Tâche] | [Tâche] | Classification 5 professions médicales | [Tâche] |
| **🤖 Modèle** | MLP + Fairness Loss | [Architecture] | [Architecture] | biomed_roberta_base frozen + RF | [Architecture] |
| **📈 Accuracy** | **97.5%** | [X.X%] | [X.X%] | **83.60 %** (D1, sans label leakage) | [X.X%] |
| **⚖️ ∆EO** | **0.025** | [X.XXX] | [X.XXX] | **0.038** (D1) | [X.XXX] |

---

## 🤖 **MODÈLES**

| **Aspect** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|------------|---------------|---------------|-----------|---------|-----------|
| **Embeddings** | DistilRoBERTa | [Type] | [Type] | biomed_roberta_base (frozen, token CLS) | [Type] |
| **Architecture** | MLP (256→128→5) | [Détail] | [Détail] | RandomForestClassifier (100 arbres, max_depth=20) | [Détail] |
| **Preprocessing** | **TF-IDF baseline**: max_features=5000, stop_words='english', lowercase=True, ngram_range=(1,2) **+ SMOTE** (genre×prof, k=5) | [Technique] | [Technique] | D1 : masquage mots-métiers → `[PROF]` + D1n : neutralisation genre (pronoms, titres) + StandardScaler | [Technique] |
| **Baseline vs Final** | 89.9% → 97.5% (+7.6%) | [X%] → [Y%] | [X%] → [Y%] | D0 88.50% → D1 83.60% (−4.9 %, label leakage supprimé) | [X%] → [Y%] |

---

## ⚖️ **FAIRNESS**

| **Métrique** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|--------------|---------------|---------------|-----------|---------|-----------|
| **Attribut Sensible** | Genre (F/M) | [Attribut] | [Attribut] | Genre (F/M) | [Attribut] |
| **∆ Demographic Parity** | 0.164 | [X.XXX] | [X.XXX] | **0.000** ✅ | [X.XXX] |
| **∆ Equality of Opportunity** | **0.025** ✅ | [X.XXX] | [X.XXX] | **0.038** ✅ (D1) | [X.XXX] |
| **Méthode Mitigation** | In-processing (λ=0.1) | [Pre/In/Post] | [Pre/In/Post] | Pre-processing (masquage lexical D1 + neutralisation genre D1n) | [Pre/In/Post] |

---

## 🛡️ **ROBUSTESSE**

| **Test** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|----------|---------------|---------------|-----------|---------|-----------|
| **Validation** | 5-fold CV | [Méthode] | [Méthode] | Split fixe 8k/1k/1k + gender swap test | [Méthode] |
| **Stabilité** | ±0.008 ✅ | [±X.XXX] | [±X.XXX] | Flip rate 12.1 % (gender swap) | [±X.XXX] |
| **Confidence Intervals** | Bootstrap (N=2000) | [Méthode/Aucune] | [Méthode/Aucune] | Aucune | [Méthode/Aucune] |

---

## 🎲 **UNCERTAINTY**

| **Aspect** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|------------|---------------|---------------|-----------|---------|-----------|
| **Méthode UQ** | ❌ Aucune vraie UQ | [BNN/MC Dropout/Aucune] | [Méthode/Aucune] | ❌ Aucune | [Méthode/Aucune] |
| **Calibration** | ❌ Non testée | [Testée/Non] | [Testée/Non] | ❌ Non testée | [Testée/Non] |

---

## 🔬 **POINTS CLÉS**

| **Dimension** | **Antoine D** | **Antoine L** | **Osman** | **Leo** | **Saad** |
|---------------|---------------|---------------|-----------|---------|-----------|
| **🏆 Principal Succès** | In-processing > Pre/Post | [Succès] | [Succès] | biomed_roberta meilleur modèle toutes versions (D0: 87.4%, D1: 82.9% Macro-F1) | [Succès] |
| **⚠️ Principale Limite** | Pas de vraie UQ | [Limite] | [Limite] | Embeddings frozen (pas de fine-tuning) + pas d'UQ | [Limite] |
| **📊 Insight Clé** | λ=0.1 optimal | [Insight] | [Insight] | D1 = meilleur compromis perf/fairness ; DP gap nul sur tous les modèles | [Insight] |

---

## 📋 **DÉTAILS PAR PROFESSIONS** (Antoine D)

| **Profession** | **% Female** | **∆DP** | **∆EO** | **Niveau de Biais** |
|----------------|---------------|---------|---------|-------------------|
| **Physician** | 50% | 0.023 | 0.023 | ✅ Très faible |
| **Dentist** | 45% | 0.153 | 0.036 | ⚠️ Moyen |
| **Psychologist** | 55% | 0.076 | 0.006 | ✅ Faible |
| **Nurse** | **72%** | **0.336** | **0.350** | 🚨 **Critique** |
| **Surgeon** | **38%** | **0.282** | **0.445** | 🚨 **Critique** |

---

## 🏆 **CLASSEMENTS ACTUELS**

### **📈 Performance**
| Rang | Projet | Accuracy | Statut |
|------|--------|----------|--------|
| 1 | Antoine D | **97.5%** | ✅ Complété |
| 2 | [Projet] | [X.X%] | [À compléter] |
| 3 | [Projet] | [X.X%] | [À compléter] |

### **⚖️ Fairness**
| Rang | Projet | ∆EO | Statut |
|------|--------|-----|--------|
| 1 | Antoine D | **0.025** | ✅ Excellent |
| 2 | [Projet] | [X.XXX] | [À compléter] |
| 3 | [Projet] | [X.XXX] | [À compléter] |

### **🛡️ Robustesse** 
| Rang | Projet | Stabilité | Statut |
|------|--------|-----------|--------|
| 1 | Antoine D | **±0.008** | ✅ Très stable |
| 2 | [Projet] | [±X.XXX] | [À compléter] |
| 3 | [Projet] | [±X.XXX] | [À compléter] |

---

## 📋 **DÉTAILS PAR RUBRIQUE (Leo)**

### 🤖 Performance par version de données (biomed_roberta_base + RF)

| **Version** | **Description** | **Accuracy** | **Macro-F1** | **Wgt-F1** |
|-------------|-----------------|--------------|--------------|------------|
| **D0** | Texte brut (label leakage présent) | 88.50 % | 87.37 % | 88.19 % |
| **D1** | Mots-métiers masqués `[PROF]` | **83.60 %** | **82.86 %** | 83.35 % |
| **D1n** | D1 + neutralisation genre | 82.90 % | 81.71 % | 82.57 % |

> D0 gonflé artificiellement : 95.5 % des bios contiennent le nom de la profession. D1 = évaluation réaliste.

---

### ⚖️ Fairness par version de données (biomed_roberta_base + RF)

| **Version** | **∆DP** | **∆EOpp** | **∆FPR** | **Acc gap** | **Interprétation** |
|-------------|---------|-----------|----------|-------------|-------------------|
| **D0** | **0.000** ✅ | 6.78 % ⚠️ | 0.43 % ✅ | 0.46 % ✅ | Label leakage masque le biais réel |
| **D1** | **0.000** ✅ | **3.79 %** ✅ | 1.15 % ✅ | 2.24 % ✅ | Meilleur compromis perf/fairness |
| **D1n** | **0.000** ✅ | 8.34 % ⚠️ | 1.34 % ✅ | 3.07 % ✅ | Neutralisation genre trop agressive |

> DP gap = 0.000 sur toutes les versions : le modèle distribue les prédictions équitablement entre genres.  
> EOpp gap minimal sur D1 (3.79 %) → version recommandée.

---

### 🛡️ Robustesse — Gender Swap (biomed_roberta_base sur D0)

| **Groupe** | **Flip rate** | **Interprétation** |
|------------|---------------|-------------------|
| Global | **12.10 %** | 1 prédiction sur 8 change avec le genre |
| Femmes | ~11 % | Légèrement moins sensibles |
| Hommes | ~13 % | Légèrement plus sensibles |

| **Métrique** | **Avant swap** | **Après swap** | **Delta** |
|--------------|---------------|----------------|-----------|
| Macro-F1 | ~82 % | ~77 % | **−4.56 pts** |

> Le modèle exploite les pronoms genrés (he/she, him/her) dans D0 → D1n atténue ce phénomène.

---

### 🔬 Comparaison modèles — D1 et D1n (version retenue)

**D1 — Mots-métiers masqués**

| **Modèle** | **Accuracy** | **Macro-F1** | **Wgt-F1** | **EOpp gap** | **FPR gap** | **Acc gap** | **Bilan** |
|------------|--------------|--------------|------------|--------------|-------------|-------------|-----------|
| **biomed_roberta_base** | **83.60 %** | **82.86 %** | 83.35 % | **3.79 %** | 1.15 % | 2.24 % | 🏆 Meilleur global |
| BiomedBERT-base-uncased | 78.40 % | 76.97 % | 77.94 % | 4.45 % | 1.30 % | 2.51 % | Bon |
| distilbert-base-uncased | 81.60 % | 80.57 % | 81.25 % | 7.25 % | 1.72 % | 4.16 % | Baseline solide |
| Bio_ClinicalBERT | 72.70 % | 72.16 % | 72.44 % | 5.66 % | 2.47 % | 6.52 % | Équitable mais moins performant |
| biobert-base-cased-v1.2 | 71.30 % | 70.29 % | 70.93 % | 9.76 % | 1.87 % | 4.12 % | ⚠️ Biais fort |
| BioM-ELECTRA-Large | 67.00 % | 66.57 % | 66.89 % | 3.72 % | 1.33 % | 1.58 % | Limité en perf |

**D1n — Mots-métiers masqués + neutralisation genre**

| **Modèle** | **Accuracy** | **Macro-F1** | **Wgt-F1** | **EOpp gap** | **FPR gap** | **Acc gap** | **Bilan** |
|------------|--------------|--------------|------------|--------------|-------------|-------------|-----------|
| **biomed_roberta_base** | **82.90 %** | **81.71 %** | 82.57 % | 8.34 % | 1.34 % | 3.07 % | 🏆 Meilleur perf |
| BiomedBERT-base-uncased | 77.50 % | 76.26 % | 77.17 % | 6.11 % | 1.36 % | 2.48 % | Bon |
| distilbert-base-uncased | 79.70 % | 78.47 % | 79.21 % | 13.60 % | 2.66 % | 7.53 % | ❌ EOpp gap très élevé |
| Bio_ClinicalBERT | 72.00 % | 70.70 % | 71.53 % | **3.92 %** | 1.72 % | 4.51 % | ✅ Plus équitable |
| biobert-base-cased-v1.2 | 67.20 % | 65.90 % | 66.81 % | 8.49 % | 1.73 % | 3.66 % | ⚠️ Chute perf |
| BioM-ELECTRA-Large | 61.80 % | 60.12 % | 61.20 % | 11.64 % | 1.28 % | 1.85 % | ❌ EOpp gap critique |

> **Synthèse** : D1 = meilleur compromis perf/fairness (biomed_roberta : 82.86 % Macro-F1, EOpp gap 3.79 %). Sur D1n, Bio_ClinicalBERT devient le plus équitable (EOpp gap 3.92 %). DistilBERT (baseline non-biomédical) est compétitif en perf (80.57 % D1) mais présente le pire EOpp gap sur D1n (13.60 %) — confirme la valeur de la spécialisation biomédicale pour la fairness. DP gap = 0.00 % pour tous.

---

*État des lieux - 24/04/2026*