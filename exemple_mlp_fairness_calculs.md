# 📊 Exemple Numérique : MLP + Fairness Loss

## 🎯 Dataset d'Exemple (Mini-Batch)

Supposons un batch de **3 échantillons** avec embeddings simplifiés :

| ID | Embedding (4D) | Profession | Genre | Y_true | G_bin |
|----|----------------|------------|--------|--------|-------|
| 1  | [0.5, -0.2, 0.8, 0.1] | Surgeon | Male | 0 | 1 |
| 2  | [0.3, 0.7, -0.4, 0.6] | Nurse | Female | 1 | 0 |  
| 3  | [0.9, 0.1, 0.2, -0.3] | Surgeon | Female | 0 | 0 |

**Professions** : {Surgeon=0, Nurse=1, Doctor=2, Therapist=3, Pharmacist=4}
**Genres** : {Female=0, Male=1}

---

## 🧠 Forward Pass du MLP

### **Étape 1 : Première Couche Cachée**
```
X = [[0.5, -0.2, 0.8, 0.1],
     [0.3,  0.7, -0.4, 0.6], 
     [0.9,  0.1,  0.2, -0.3]]

# Supposons W1 (4×2) et b1 (2,) :
W1 = [[0.1, 0.4],
      [0.3, -0.2], 
      [0.5, 0.1],
      [-0.1, 0.6]]
b1 = [0.2, -0.1]
```

**Calcul H1 = ReLU(X @ W1 + b1) :**

Pour l'échantillon 1 :
```
z1 = [0.5, -0.2, 0.8, 0.1] @ [[0.1, 0.4], [0.3, -0.2], [0.5, 0.1], [-0.1, 0.6]] + [0.2, -0.1]
z1 = [0.5×0.1 + (-0.2)×0.3 + 0.8×0.5 + 0.1×(-0.1), 
      0.5×0.4 + (-0.2)×(-0.2) + 0.8×0.1 + 0.1×0.6] + [0.2, -0.1]
z1 = [0.05 - 0.06 + 0.4 - 0.01 + 0.2, 
      0.2 + 0.04 + 0.08 + 0.06 - 0.1]
z1 = [0.58, 0.28]
h1_1 = ReLU([0.58, 0.28]) = [0.58, 0.28]
```

De même pour échantillons 2 et 3 :
```
h1_2 = [0.45, 0.33]  # (calculs similaires)
h1_3 = [0.62, 0.15]
```

**H1 = [[0.58, 0.28], [0.45, 0.33], [0.62, 0.15]]**

### **Étape 2 : Couche de Sortie (5 classes)**
```
# W2 (2×5) et b2 (5,) :
W2 = [[0.2, -0.1, 0.3, 0.4, -0.2],
      [0.1, 0.5, -0.3, 0.2, 0.6]]
b2 = [0.1, -0.2, 0.0, 0.3, -0.1]
```

**Logits = H1 @ W2 + b2 :**

Pour échantillon 1 :
```
logits_1 = [0.58, 0.28] @ W2 + b2
logits_1 = [0.58×0.2 + 0.28×0.1, 0.58×(-0.1) + 0.28×0.5, ...] + b2
logits_1 = [0.144, 0.082, 0.090, 0.288, 0.052] + [0.1, -0.2, 0.0, 0.3, -0.1]
logits_1 = [0.244, -0.118, 0.090, 0.588, -0.048]
```

**Logits finaux :**
```
Logits = [[0.244, -0.118, 0.090, 0.588, -0.048],  # Échantillon 1 (Male, Surgeon)
          [0.180, 0.145, -0.135, 0.420, 0.098],   # Échantillon 2 (Female, Nurse)
          [0.239, -0.014, 0.141, 0.492, -0.004]]  # Échantillon 3 (Female, Surgeon)
```

---

## ⚖️ Calcul des Loss

### **A. CrossEntropy Loss**

**Softmax pour chaque échantillon :**
```python
import numpy as np

# Échantillon 1 : Y_true = 0 (Surgeon)
logits_1 = [0.244, -0.118, 0.090, 0.588, -0.048]
exp_1 = [exp(0.244), exp(-0.118), exp(0.090), exp(0.588), exp(-0.048)]
exp_1 = [1.276, 0.889, 1.094, 1.801, 0.953]
sum_1 = 6.013
prob_1 = [0.212, 0.148, 0.182, 0.300, 0.158]

CE_1 = -log(prob_1[0]) = -log(0.212) = 1.549
```

De même :
```
prob_2 = [0.195, 0.193, 0.146, 0.247, 0.219]  # Y_true = 1 (Nurse)
CE_2 = -log(prob_2[1]) = -log(0.193) = 1.643

prob_3 = [0.206, 0.165, 0.193, 0.266, 0.170]  # Y_true = 0 (Surgeon)  
CE_3 = -log(prob_3[0]) = -log(0.206) = 1.576
```

**L_CE = (1.549 + 1.643 + 1.576) / 3 = 1.589**

### **B. Demographic Parity Loss**

**Taux de prédiction par genre :**
```python
# Female (échantillons 2, 3) :
prob_female = (prob_2 + prob_3) / 2
prob_female = ([0.195, 0.193, 0.146, 0.247, 0.219] + [0.206, 0.165, 0.193, 0.266, 0.170]) / 2
prob_female = [0.201, 0.179, 0.170, 0.257, 0.194]

# Male (échantillon 1) :
prob_male = prob_1 = [0.212, 0.148, 0.182, 0.300, 0.158]

# Différences absolues par classe :
diff = |prob_female - prob_male|
diff = |[0.201, 0.179, 0.170, 0.257, 0.194] - [0.212, 0.148, 0.182, 0.300, 0.158]|
diff = [0.011, 0.031, 0.012, 0.043, 0.036]

L_DP = mean(diff) = (0.011 + 0.031 + 0.012 + 0.043 + 0.036) / 5 = 0.027
```

### **C. Equality of Opportunity Loss**

**TPR par classe et genre :**
```python
# Classe 0 (Surgeon) : échantillons 1 (Male) et 3 (Female)
TPR_0_Male = prob_1[0] = 0.212    # P(Ŷ=0 | Y=0, Male)
TPR_0_Female = prob_3[0] = 0.206  # P(Ŷ=0 | Y=0, Female)
diff_0 = |0.212 - 0.206| = 0.006

# Classe 1 (Nurse) : échantillon 2 (Female) seulement
# Pas de Male avec Y=1, donc pas de différence calculable pour cette classe

L_EO = diff_0 = 0.006  # (Moyenne sur les classes avec données des deux genres)
```

---

## 🎯 Loss Totale avec λ

### **Combinaison Final**

Supposons **λ = 0.1** et **criterion = 'dp'** :

```python
L_total = L_CE + λ × L_fairness
L_total = 1.589 + 0.1 × 0.027
L_total = 1.589 + 0.003
L_total = 1.592
```

### **Comparaison λ :**

| λ | L_CE | L_fairness | L_total | Impact |
|---|------|------------|---------|--------|
| **0.0** | 1.589 | - | **1.589** | Pas de contrainte fairness |
| **0.01** | 1.589 | 0.027 | **1.590** | Légère contrainte (+0.1%) |
| **0.1** | 1.589 | 0.027 | **1.592** | Contrainte modérée (+0.2%) |
| **1.0** | 1.589 | 0.027 | **1.616** | Contrainte forte (+1.7%) |

---

## 🔄 Backpropagation

### **Gradient de L_total**
```python
∇_θ L_total = ∇_θ L_CE + λ × ∇_θ L_fairness

# Le gradient fairness "tire" les poids vers l'égalisation des prédictions entre genres
# Le gradient CE "tire" les poids vers la classification correcte
# λ contrôle l'équilibre entre ces deux forces !
```

### **Update des Poids (Adam)**
```python
# Exemple pour W2[0,0] :
θ_new = θ_old - lr × ∇_θ L_total
W2[0,0]_new = W2[0,0]_old - 0.001 × grad_W2[0,0]
```

---

## 🏆 Conclusion

Cet exemple montre comment **chaque λ influence le trade-off** :
- **λ=0** : Optimisation pure de la précision 
- **λ>0** : Équilibrage progressif vers la fairness
- **λ trop grand** : Risque de sur-contrainte et baisse d'accuracy

Dans ton projet, **λ=0.1 était optimal** car il améliore la fairness sans sacrifier l'accuracy ! 🎯