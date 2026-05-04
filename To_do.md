# TODO LIST - ANALYSE DE BIAIS NLP MÉDICAL

## ✅ SETUP INITIAL ET BASELINE

### Étape 1: Organisation du projet
- [ ] Créer dossier `experiments/`
- [ ] Créer dossier `results/`
- [ ] Créer dossier `utils/`
- [ ] Initialiser git repository
- [ ] Setup MLflow: lancer `mlflow ui --port 5000`

### Étape 2: Finaliser le baseline actuel
- [ ] Copier le code actuel de fairness.ipynb dans `01_baseline_performance.ipynb`
- [ ] Nettoyer et documenter chaque section
- [ ] Ajouter des commentaires explicatifs
- [ ] Calculer métriques complètes sur test set
- [ ] Sauvegarder le modèle baseline avec pickle
- [ ] Logger les résultats baseline dans MLflow

## 🔍 PHASE 1: DÉTECTION ET QUANTIFICATION DES BIAIS

### Étape 3: Implémenter métriques de fairness
- [ ] Créer `02_bias_detection.ipynb`
- [ ] Récupérer les genres du test set depuis `df_test['gender']`
- [ ] Implémenter fonction `calculate_demographic_parity()`
- [ ] Implémenter fonction `calculate_equality_opportunity()`
- [ ] Implémenter fonction `calculate_equalized_odds()`
- [ ] Calculer toutes les métriques pour le modèle baseline

### Étape 4: Matrices de confusion par genre
- [ ] Créer matrices de confusion séparées par genre (comme vos résultats précédents)
- [ ] Calculer TPR, FPR, PPV pour chaque genre
- [ ] Créer visualisations heatmap des matrices
- [ ] Calculer les écarts entre groupes (∆DP, ∆EO, ∆Equalized Odds)

### Étape 5: Analyse intersectionnelle
- [ ] Analyser biais par Genre × Profession (dentist, nurse, physician, psychologist, surgeon)
- [ ] Créer sous-groupes: female_nurse, male_nurse, female_physician, etc.
- [ ] Calculer métriques fairness pour chaque sous-groupe
- [ ] Identifier les discriminations multiples les plus importantes

### Étape 6: Tests statistiques
- [ ] Implémenter test Chi2 pour indépendance genre/prédictions
- [ ] Calculer intervalles de confiance avec bootstrap
- [ ] Tester significativité des écarts observés
- [ ] Documenter la robustesse statistique des biais détectés

## 🧬 PHASE 2: VARIATIONS DU MODÈLE

### Étape 7: Comparaison modèles d'embedding
- [ ] Créer `03_embedding_models_comparison.ipynb`
- [ ] Adapter fonction `get_roberta_embeddings()` pour `bert-base-uncased`
- [ ] Adapter fonction pour `roberta-base`
- [ ] Adapter fonction pour `dmis-lab/biobert-v1.1`
- [ ] Adapter fonction pour `emilyalsentzer/Bio_ClinicalBERT`
- [ ] Extraire embeddings pour test set seulement (économie de temps)
- [ ] Entraîner même RF classifier sur chaque type d'embedding
- [ ] Calculer fairness metrics pour chaque modèle
- [ ] Créer tableau comparatif final

### Étape 8: Optimisation hyperparamètres Random Forest
- [ ] Créer `04_rf_hyperparameter_tuning.ipynb`
- [ ] Tester n_estimators: 50, 100, 200
- [ ] Tester max_depth: 10, 20, None
- [ ] Tester min_samples_split: 2, 5, 10
- [ ] Évaluer impact sur accuracy ET fairness pour chaque configuration
- [ ] Sélectionner meilleur compromis accuracy/fairness

### Étape 9: Comparaison algorithmes de classification
- [ ] Créer `05_classifier_comparison.ipynb`
- [ ] Tester Logistic Regression sur embeddings DistilRoBERTa
- [ ] Tester SVM avec kernel RBF
- [ ] Tester XGBoost Classifier
- [ ] Tester MLP Neural Network
- [ ] Calculer fairness metrics pour chaque algorithme
- [ ] Identifier l'algorithme naturellement le plus équitable

## ⚖️ PHASE 3: ANALYSE APPROFONDIE DE FAIRNESS

### Étape 10: Feature importance analysis
- [ ] Créer `06_feature_importance_analysis.ipynb`
- [ ] Analyser `rf_classifier.feature_importances_` sur les 768 dimensions
- [ ] Identifier top 50 features les plus discriminantes
- [ ] Visualiser importance par dimension d'embedding
- [ ] Corréler avec activations moyennes par genre/profession
- [ ] Identifier quelles dimensions créent les biais

### Étape 11: Distribution analysis
- [ ] Créer `07_prediction_distribution_analysis.ipynb`
- [ ] Analyser probabilités prédites par groupe démographique
- [ ] Créer histogrammes des scores par genre/profession
- [ ] Calculer moyennes et variances par groupe
- [ ] Détecter patterns systématiques dans les distributions
- [ ] Identifier si certains groupes ont des scores biaisés

### Étape 12: Métriques de fairness complètes
- [ ] Créer `08_comprehensive_fairness_metrics.ipynb`
- [ ] Implémenter Calibration/Predictive Parity
- [ ] Calculer intervalles de confiance pour toutes métriques (bootstrap)
- [ ] Créer courbes ROC séparées par groupe
- [ ] Tester différents seuils de décision
- [ ] Documenter toutes les violations de fairness

## 🔧 PHASE 4: TECHNIQUES DE MITIGATION

### Étape 13: Pre-processing debiasing
- [ ] Créer `09_preprocessing_mitigation.ipynb`
- [ ] Implémenter resampling (SMOTE sur embeddings)
- [ ] Implémenter undersampling des groupes majoritaires
- [ ] Implémenter class weighting dans Random Forest
- [ ] Tester data augmentation par ajout de bruit aux embeddings minoritaires
- [ ] Mesurer impact de chaque technique sur fairness ET accuracy
- [ ] Comparer efficacité relative des méthodes

### Étape 14: In-processing fairness
- [ ] Créer `10_inprocessing_fairness.ipynb`
- [ ] Implémenter custom loss function avec terme fairness
- [ ] Tester différents λ (0.01, 0.1, 0.5, 1.0) pour trade-off
- [ ] Implémenter pénalité pour demographic parity dans loss
- [ ] Implémenter pénalité pour equality of opportunity
- [ ] Évaluer trade-off accuracy vs fairness pour chaque λ

### Étape 15: Post-processing fairness
- [ ] Créer `11_postprocessing_fairness.ipynb`
- [ ] Implémenter optimisation de seuils pour demographic parity
- [ ] Implémenter optimisation de seuils pour equality of opportunity
- [ ] Trouver seuils optimaux par genre/profession
- [ ] Mesurer dégradation de l'accuracy globale
- [ ] Évaluer feasibility pratique des seuils différenciés

## 📊 PHASE 5: ÉVALUATION ET SYNTHÈSE

### Étape 16: Comparative analysis
- [ ] Créer `12_comparative_analysis.ipynb`
- [ ] Compiler tous les résultats dans tableau unifié
- [ ] Créer radar plots pour visualiser trade-offs
- [ ] Calculer scores composites accuracy+fairness
- [ ] Identifier configurations optimales par use case
- [ ] Analyser robustesse des solutions proposées

### Étape 17: Robustness testing
- [ ] Créer `13_robustness_testing.ipynb`
- [ ] Cross-validation des métriques fairness (5-fold)
- [ ] Bootstrap confidence intervals pour toutes métriques
- [ ] Test sur sous-échantillons de différentes tailles
- [ ] Analyser sensibilité aux hyperparamètres optimaux
- [ ] Tester stabilité temporelle (si dates disponibles)

### Étape 18: Final report et recommandations
- [ ] Créer `14_final_report.ipynb`
- [ ] Synthèse exécutive des découvertes principales
- [ ] Recommandations techniques pour déploiement clinique
- [ ] Discussion des limites et biais résiduels non résolus
- [ ] Propositions pour recherches futures
- [ ] Guidelines pratiques pour usage responsable

## 🛠️ CRÉATION D'OUTILS TRANSVERSAUX

### Étape 19: Utils library
- [ ] Créer `utils/fairness_metrics.py`
- [ ] Implémenter classe `FairnessAnalyzer` complète
- [ ] Créer fonctions de visualisation standardisées
- [ ] Implémenter générateur de rapports automatique
- [ ] Créer fonctions de comparaison de modèles

### Étape 20: MLflow integration
- [ ] Créer `utils/mlflow_helper.py`
- [ ] Implémenter logging automatique des expériences
- [ ] Standardiser format des métriques loggées
- [ ] Créer dashboard de comparaison des modèles
- [ ] Setup tracking des hyperparamètres et résultats

## 📋 FINALISATION ET DOCUMENTATION

### Étape 21: Code cleanup et documentation
- [ ] Nettoyer tous les notebooks créés
- [ ] Ajouter docstrings à toutes les fonctions
- [ ] Créer README.md avec instructions d'utilisation
- [ ] Créer requirements.txt avec toutes dépendances
- [ ] Versionner tout le code avec git

### Étape 22: Validation finale
- [ ] Reproduire tous les résultats from scratch
- [ ] Vérifier cohérence des métriques entre notebooks
- [ ] Valider que toutes conclusions sont supportées
- [ ] Test de la reproductibilité sur machine différente
- [ ] Backup final de tous résultats et modèles

## 🎯 OBJECTIFS DE SORTIE

À la fin, vous devez pouvoir répondre à:
- [ ] Quel est le niveau exact de biais dans votre modèle baseline?
- [ ] Quel embedding/classifier est le plus équitable?
- [ ] Quelles techniques de mitigation sont les plus efficaces?
- [ ] Quel est le trade-off optimal accuracy vs fairness?
- [ ] Quelles sont vos recommandations pour usage clinique?

**Estimation totale**: 170h sur 6-7 semaines
**Livrables**: 14 notebooks + utils + rapport final + dashboard MLflow