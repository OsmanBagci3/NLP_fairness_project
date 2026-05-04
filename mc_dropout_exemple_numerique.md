# 🎲 MC Dropout : Exemple Numérique Concret

## 🎯 Ce que tu fais ACTUELLEMENT

### **Forward Pass Standard (model.eval())**
```python
# Ton code actuel
model.eval()  # ← Dropout DÉSACTIVÉ
logits = model(x_test)
probs = softmax(logits)
```

**Résultat pour 1 échantillon :**
```
# Une seule prédiction déterministe
Probs = [0.212, 0.148, 0.182, 0.300, 0.158]  # Surgeon, Nurse, Doctor, Therapist, Pharmacist
Predicted = 3 (Therapist)  # argmax
Confidence = 0.300  # max probability
```

---

## 🎲 Ce que tu POURRAIS faire : MC Dropout

### **Multiple Forward Passes (model.train())**
```python
# MC Dropout - Nouvelle approche
model.train()  # ← Dropout ACTIF pendant l'inférence !
T = 5  # Nombre de passes

predictions = []
for t in range(T):
    logits_t = model(x_test)  # Chaque passe = masques dropout différents
    probs_t = softmax(logits_t)
    predictions.append(probs_t)
```

### **Exemple Numérique avec T=5 passes**

**Échantillon : Female Surgeon (potentiellement difficile à classifier)**

| Passe | Dropout Mask H1 | Dropout Mask H2 | Probabilités | Prédiction |
|-------|-----------------|-----------------|--------------|------------|
| **1** | [1, 0, 1] | [1, 1] | [0.45, 0.12, 0.15, 0.20, 0.08] | **Surgeon** |
| **2** | [1, 1, 0] | [0, 1] | [0.22, 0.18, 0.25, 0.28, 0.07] | **Therapist** |
| **3** | [0, 1, 1] | [1, 0] | [0.38, 0.15, 0.18, 0.22, 0.07] | **Surgeon** |
| **4** | [1, 0, 0] | [1, 1] | [0.30, 0.20, 0.22, 0.18, 0.10] | **Surgeon** |
| **5** | [0, 0, 1] | [0, 1] | [0.15, 0.25, 0.30, 0.22, 0.08] | **Doctor** |

### **Calcul de l'Incertitude**

```python
# 1. Prédiction Moyenne (Predictive Mean)
mean_probs = np.mean(predictions, axis=0)
mean_probs = [0.30, 0.18, 0.22, 0.22, 0.08]

# 2. Incertitude Épistémique (Epistemic Uncertainty)
variance = np.var(predictions, axis=0)
epistemic_uncertainty = np.sum(variance)  # Total uncertainty
epistemic_uncertainty = 0.0125  # Exemple

# 3. Entropie Prédictive (Predictive Entropy)  
entropy = -np.sum(mean_probs * np.log(mean_probs))
entropy = 1.52  # Plus élevée = plus incertain
```

---

## ⚖️ Détection de Biais dans l'Incertitude

### **Comparaison Male vs Female Surgeons**

| Genre | Échantillon | Mean Prob(Surgeon) | Épistémique Uncertainty | Entropie | Interprétation |
|-------|-------------|-------------------|------------------------|----------|----------------|
| **Male** | Male Surgeon | **0.65** | **0.008** | **1.12** | Confiant ✅ |
| **Female** | Female Surgeon | **0.30** | **0.025** | **1.52** | Incertain ⚠️ |

### **Biais d'Incertitude Détecté !**
```python
# Le modèle est 3x plus incertain sur les femmes chirurgiennes !
uncertainty_bias = 0.025 / 0.008 = 3.1x

# Ceci révèle un biais INVISIBLE à ∆EO :
# - ∆EO mesure la différence de TPR entre genres
# - Mais ne capture pas la "confiance" différentielle du modèle
```

---

## 🔬 Implémentation Pratique

### **Code Minimal pour ton Notebook**
```python
def mc_dropout_prediction(model, x, n_samples=10):
    """MC Dropout pour mesurer l'incertitude."""
    model.train()  # ← Clé : garder dropout actif !
    
    predictions = []
    for _ in range(n_samples):
        with torch.no_grad():
            logits = model(x)
            probs = F.softmax(logits, dim=1)
            predictions.append(probs.cpu().numpy())
    
    predictions = np.array(predictions)
    
    # Statistiques
    mean_pred = predictions.mean(axis=0)
    var_pred = predictions.var(axis=0)
    entropy = -np.sum(mean_pred * np.log(mean_pred + 1e-8), axis=1)
    
    return {
        'mean_prediction': mean_pred,
        'epistemic_uncertainty': var_pred.sum(axis=1), 
        'predictive_entropy': entropy
    }

# Usage sur ton dataset
results_male = mc_dropout_prediction(model, male_surgeons_embeddings)
results_female = mc_dropout_prediction(model, female_surgeons_embeddings)

print(f"Male Surgeon Uncertainty: {results_male['epistemic_uncertainty'].mean():.4f}")
print(f"Female Surgeon Uncertainty: {results_female['epistemic_uncertainty'].mean():.4f}")
```

### **Métriques de Biais d'Incertitude**
```python
# Nouvelles métriques que tu pourrais calculer
uncertainty_ratio = female_uncertainty.mean() / male_uncertainty.mean()
confidence_gap = male_confidence.mean() - female_confidence.mean()

print(f"📊 Uncertainty Bias Ratio: {uncertainty_ratio:.2f}")
print(f"📊 Confidence Gap: {confidence_gap:.3f}")
```

---

## 🎯 Pourquoi c'est Important pour la Fairness

### **Biais d'Incertitude vs Biais de Prédiction**

| Métrique | Male Surgeon | Female Surgeon | Interpretation |
|----------|--------------|----------------|----------------|
| **Accuracy** | 92% | 90% | Légère différence |
| **∆EO** | - | 0.025 | Acceptable |
| **Épistémique Unc.** | 0.008 | **0.025** | **Biais fort !** |
| **Entropie** | 1.12 | **1.52** | **Moins confiant** |

### **Impact Pratique**
- **Système médical** : Le modèle sera moins "sûr" quand il recommande une femme chirurgienne
- **Prise de décision** : Biais implicite dans la confiance des prédictions
- **Équité** : Même accuracy mais traitement différent de l'incertitude

---

## 🚀 Ta Prochaine Expérimentation

1. **Modifie ton Notebook 10** : Ajoute `mc_dropout_prediction()`
2. **Compare les incertitudes** par genre et profession  
3. **Nouvelle métrique** : "Uncertainty Fairness" 
4. **Découvre** si λ=0.1 réduit aussi le biais d'incertitude !

**Une ligne de code pour révéler un biais invisible** ! 🎲✨

---

# 🚀 3 Techniques UQ Avancées pour ton Projet

## 🎭 **1. Deep Ensembles - "L'Équipe de Modèles"**

### **🧠 Principe**
Au lieu d'1 seul FairMLP, tu entraînes **5 FairMLPs indépendants** avec des graines aléatoires différentes :

```python
# Ce que tu fais actuellement
model = FairMLP(768, [128, 64], 5, dropout=0.2)
model = train_fair_mlp(model, λ=0.1)

# Deep Ensembles
ensemble_models = []
for seed in [42, 123, 456, 789, 999]:
    torch.manual_seed(seed)
    model_i = FairMLP(768, [128, 64], 5, dropout=0.2)
    model_i = train_fair_mlp(model_i, λ=0.1)  # Même λ, seed différente
    ensemble_models.append(model_i)
```

### **📊 Exemple Numérique**
**Prédiction pour "Female Surgeon" :**

| Modèle | Seed | Prob(Surgeon) | Prédiction | Convergence |
|--------|------|---------------|------------|-------------|
| **M1** | 42 | **0.72** | Surgeon | Confiant ✅ |
| **M2** | 123 | **0.58** | Surgeon | Moyennement confiant |
| **M3** | 456 | **0.41** | Therapist | **Désaccord !** ⚠️ |
| **M4** | 789 | **0.65** | Surgeon | Confiant ✅ |
| **M5** | 999 | **0.33** | Therapist | **Désaccord !** ⚠️ |

### **🔍 Analyse du Disagreement**
```python
predictions = [0.72, 0.58, 0.41, 0.65, 0.33]
ensemble_mean = np.mean(predictions)  # 0.538 → Surgeon
ensemble_std = np.std(predictions)    # 0.165 → FORTE incertitude !

# Disagreement = mesure de biais épistémique
disagreement_female = 0.165  # Forte variance
disagreement_male = 0.045    # Faible variance

bias_ratio = 0.165 / 0.045 = 3.67x  # Le modèle "n'arrive pas à se décider" sur les femmes !
```

### **⚖️ Applications Fairness**
- **Détection** : Quelles combinaisons genre×profession font le plus "débat" entre modèles ?
- **Robustesse** : L'amélioration fairness (λ=0.1) est-elle **reproductible** across seeds ?

---

## ⚡ **2. Weak Ensembles - "L'Efficacité Intelligente"**

### **🧠 Principe**  
Même bénéfice que Deep Ensembles mais **5× moins cher** !

#### **A. BatchEnsemble**
```python
# Au lieu de 5 modèles complets, on fait :
class FairMLPBatchEnsemble(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, n_ensemble=5):
        self.W_base = nn.Linear(input_dim, hidden_dims[0])  # Poids partagés
        self.r_vectors = nn.Parameter(torch.randn(n_ensemble, input_dim))     # Vecteurs rapides
        self.s_vectors = nn.Parameter(torch.randn(n_ensemble, hidden_dims[0])) # Vecteurs rapides
        
    def forward(self, x, ensemble_id=0):
        # W_effective = W_base ⊙ (r_i ⊗ s_i^T)  ← Factorisation rang-1 !
        r_i = self.r_vectors[ensemble_id]
        s_i = self.s_vectors[ensemble_id]
        W_eff = self.W_base.weight * torch.outer(s_i, r_i)
        return F.linear(x, W_eff, self.W_base.bias)
```

**Avantage** : 5× moins de paramètres qu'un vrai ensemble !

#### **B. MIMO (Multiple Input Multiple Output)**
```python
class FairMLPMIMO(nn.Module):
    def forward(self, x):
        # 1 forward pass → 5 prédictions indépendantes !
        h = self.shared_layers(x)
        outputs = []
        for head in self.ensemble_heads:
            outputs.append(head(h))
        return torch.stack(outputs)  # [5, batch_size, n_classes]
```

### **📊 Comparaison Coût/Bénéfice**

| Méthode | Paramètres | Temps d'Entraînement | Qualité UQ | Idéal pour |
|---------|------------|---------------------|-------------|------------|
| **Deep Ensembles** | 5× plus | 5× plus | ⭐⭐⭐⭐⭐ | Recherche/Analyse |
| **BatchEnsemble** | 1.2× plus | 1.1× plus | ⭐⭐⭐⭐ | Production |
| **MIMO** | 1× plus | 1× plus | ⭐⭐⭐ | Temps réel |

---

## 🎲 **3. BNN Post-Hoc - "La Conversion Bayésienne"**

### **🧠 Principe**
Tu prends ton FairMLP **déjà entraîné** et tu le convertis en réseau bayésien **sans ré-entraînement** !

### **🔧 Implémentation**
```python
# Ton modèle actuel
class FairMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, n_classes, dropout=0.3):
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers += [nn.Linear(prev, h), nn.ReLU(), nn.Dropout(dropout)]
            prev = h
        layers.append(nn.Linear(prev, n_classes))
        self.net = nn.Sequential(*layers)

# Version Bayésienne Post-Hoc
class FairMLPBayesian(nn.Module):
    def __init__(self, pretrained_model):
        super().__init__()
        self.net = pretrained_model.net  # Copie les poids pré-entraînés !
        
        # Remplacement : Dropout → Bayesian Normalization
        for i, layer in enumerate(self.net):
            if isinstance(layer, nn.Dropout):
                self.net[i] = BayesianNormalizationLayer()

class BayesianNormalizationLayer(nn.Module):
    def forward(self, h):
        # BNL(h) = (h - μ̂) / σ̂ × γ(1 + ε) + β
        # ε ~ N(0, I)  ← Injection de bruit aléatoire !
        
        mu = h.mean(dim=0, keepdim=True)
        sigma = h.std(dim=0, keepdim=True) + 1e-5
        normalized = (h - mu) / sigma
        
        gamma = nn.Parameter(torch.ones_like(sigma))
        beta = nn.Parameter(torch.zeros_like(mu))
        epsilon = torch.randn_like(h) * 0.1  # Bruit bayésien
        
        return normalized * gamma * (1 + epsilon) + beta
```

### **⚡ Utilisation**
```python
# 1. Prendre ton modèle optimisé existant
trained_model = load_model('best_fairmlp_lambda_01.pth')

# 2. Conversion bayésienne (0 ré-entraînement !)
bayesian_model = FairMLPBayesian(trained_model)

# 3. Sampling bayésien instantané
uncertainty_samples = []
for _ in range(20):
    pred = bayesian_model(x_test)  # Chaque passe → prédiction différente !
    uncertainty_samples.append(pred)

epistemic_uncertainty = torch.stack(uncertainty_samples).var(0)
```

---

## 🎯 **Quelle Technique Choisir pour ton Projet ?**

| **Objectif** | **Technique Recommandée** | **Effort** | **Gain Fairness** |
|-------------|--------------------------|------------|-------------------|
| **🚀 Proof of Concept** | **MC Dropout** | Minimal | Détection biais incertitude |
| **📊 Analyse Robuste** | **Deep Ensembles** | Élevé | Reproducibilité fairness |
| **⚡ Efficacité** | **BatchEnsemble** | Moyen | Compromise coût/qualité |
| **🎲 Innovation** | **BNN Post-Hoc** | Faible | Incertitude bayésienne complète |

### **🏆 Ma Recommandation :**
1. **Start** : MC Dropout (1 ligne de code)
2. **Explore** : Deep Ensembles (5 seeds de ton λ=0.1 optimal)
3. **Scale** : BatchEnsemble si tu veux passer aux transformers

**Chaque technique révèle une facette différente du biais d'incertitude !** 🎯✨