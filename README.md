# 🎯 Système de Classement ATS des Candidats

Un système d'IA sophistiqué basé sur la théorie des graphes pour classer les candidats à l'emploi en utilisant des algorithmes PageRank avancés. Cet outil aide les professionnels des RH et les recruteurs à prendre des décisions d'embauche basées sur les données en faisant correspondre intelligemment les candidats aux exigences du poste.

![Python](https://img.shields.io/badge/python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-1.28+-red.svg)
![NetworkX](https://img.shields.io/badge/networkx-3.0+-green.svg)

## 📋 Table des Matières

- [Aperçu](#aperçu)
- [Fonctionnalités](#fonctionnalités)
- [Fonctionnement](#fonctionnement)
- [Fondements Mathématiques](#fondements-mathématiques)
- [Installation](#installation)
- [Utilisation](#utilisation)
- [Formats de Fichiers](#formats-de-fichiers)
- [Options de Configuration](#options-de-configuration)
- [Exemple de Flux de Travail](#exemple-de-flux-de-travail)
- [Architecture](#architecture)
- [Référence API](#référence-api)
- [Contribuer](#contribuer)
- [Licence](#licence)

## 🌟 Aperçu

Le Système de Classement ATS des Candidats utilise une **approche basée sur les graphes** pour modéliser les relations entre les exigences du poste, les compétences et les candidats. En appliquant l'algorithme PageRank (le même algorithme qui alimente les classements de recherche de Google), le système fournit des évaluations de candidats nuancées et multidimensionnelles qui vont au-delà de la simple correspondance de mots-clés.

### Avantages Clés

- **Évaluation Holistique** : Prend en compte la maîtrise des compétences, la pondération d'importance et les années d'expérience
- **Pénalités Naturelles pour les Lacunes** : Les compétences manquantes réduisent automatiquement les classements sans notation manuelle
- **Résultats Explicables** : Des décompositions détaillées montrent exactement pourquoi chaque candidat a été classé à sa position
- **Configuration Flexible** : Plusieurs modes pour intégrer l'expérience et une pondération personnalisable
- **Support Multi-Postes** : Classez les candidats pour plusieurs postes simultanément

## ✨ Fonctionnalités

### Capacités Principales

- **Classement Basé sur les Graphes** : Utilise l'algorithme PageRank pour faire circuler l'importance depuis les exigences du poste à travers les compétences jusqu'aux candidats
- **Intégration de l'Expérience** : Quatre modes pour incorporer les années d'expérience (boost, direct, both, none)
- **Correspondance des Compétences** : Correspondance intelligente qui considère à la fois la maîtrise et l'importance
- **Analyse des Lacunes** : Identification automatique des compétences critiques manquantes
- **Classements Multi-Postes** : Support pour classer les candidats pour plusieurs postes en une seule exécution

### Interface Utilisateur

- **Tableau de Bord Streamlit Interactif** : Interface web belle et intuitive
- **Visualisations en Temps Réel** : Graphiques radar, graphiques à barres et cartes de couverture des compétences
- **Profils de Candidats Détaillés** : Décompositions approfondies pour chaque candidat
- **Formats d'Export Multiples** :
  - CSV Basique (aperçu rapide)
  - CSV Détaillé (analyse complète avec recommandations RH)
  - Rapport PDF Professionnel (résumé exécutif + profils détaillés)

### Fonctionnalités d'Export

Le système génère trois types de rapports :

1. **CSV Basique** : Tableau de classement rapide avec les métriques essentielles
2. **CSV Détaillé** : Analyse complète adaptée aux RH incluant :
   - Recommandations de priorité d'embauche
   - Évaluation des besoins de formation
   - Décompositions détaillées des compétences
   - Évaluation de l'expérience
3. **Rapport PDF Professionnel** : Document prêt pour publication avec :
   - Résumé exécutif
   - Aperçu visuel du classement
   - Profils de candidats détaillés
   - Recommandations RH
   - Annexe méthodologique

## 🔧 Fonctionnement

### Le Modèle de Graphe

Le système construit un graphe orienté avec quatre types de nœuds :

1. **Nœud Poste** : Représente l'offre d'emploi
2. **Nœuds Compétences** : Compétences individuelles requises pour le poste
3. **Nœuds Candidats** : Chaque candidat
4. **Nœud Candidat Parfait** : Candidat théorique idéal pour la normalisation

### Processus de Classement

```
Exigences du Poste → Compétences (pondérées par importance) → Candidats (pondérés par maîtrise × expérience)
```

1. **Construction du Graphe** :
   - Le poste se connecte aux compétences avec des poids d'importance
   - Les compétences se connectent aux candidats avec des poids de maîtrise
   - Les modificateurs d'expérience ajustent les poids des arêtes

2. **Calcul PageRank** :
   - La marche aléatoire commence depuis le nœud poste
   - Le flux se distribue à travers les compétences vers les candidats
   - Les scores finaux représentent la qualité globale de correspondance

3. **Normalisation** :
   - Scores normalisés par rapport au "candidat parfait"
   - Produit une échelle interprétable de 0-100

4. **Génération d'Explication** :
   - Tracer les contributions de chaque compétence
   - Identifier les lacunes et les forces
   - Calculer l'impact de l'expérience

## 📐 Fondements Mathématiques

### Théorie des Graphes

Le système repose sur un **graphe orienté pondéré** G = (V, E, w) où :

- **V** : Ensemble des sommets (nœuds)
  - V = V_job ∪ V_skills ∪ V_candidates ∪ {v_perfect}
  - |V| = 1 + n_skills + n_candidates + 1

- **E** : Ensemble des arêtes orientées
  - E ⊆ V × V
  - Chaque arête (u, v) ∈ E a un poids w(u, v) ∈ ℝ⁺

- **w** : Fonction de pondération
  - w : E → ℝ⁺
  - w(u, v) représente la force de la connexion de u vers v

### Structure du Graphe

#### 1. Arêtes Poste → Compétences

Pour chaque compétence s_i requise :

```
w(v_job, s_i) = importance(s_i) ∈ [0, 1]
```

Où `importance(s_i)` est l'importance relative de la compétence pour le poste.

#### 2. Arêtes Compétences → Candidats

Pour chaque candidat c_j possédant la compétence s_i :

```
w(s_i, c_j) = proficiency(c_j, s_i) × boost_exp(c_j)
```

Où :
- `proficiency(c_j, s_i) ∈ [0, 1]` : niveau de maîtrise du candidat
- `boost_exp(c_j) ∈ [0.5, 1.5]` : facteur de boost basé sur l'expérience

#### 3. Fonction de Boost d'Expérience

Le facteur de boost d'expérience est calculé comme suit :

```
boost_exp(y_cand, y_req, y_pref) = 
  ⎧ 0.5 + 0.5 × (y_cand / y_req)              si y_cand < y_req
  ⎪ 1.0 + 0.3 × ((y_cand - y_req) / Δy)      si y_req ≤ y_cand ≤ y_pref
  ⎨ 1.3 - 0.2 × (min(y_cand - y_pref, y_pref) / y_pref)
  ⎩                                            si y_cand > y_pref
```

Où :
- y_cand : années d'expérience du candidat
- y_req : années requises
- y_pref : années préférées
- Δy = y_pref - y_req

**Interprétation** :
- Si expérience insuffisante : pénalité de 50% à 100%
- Si expérience dans la plage souhaitée : bonus jusqu'à 130%
- Si expérience excessive : rendements décroissants (130% à 150%)

### Algorithme PageRank

L'algorithme PageRank calcule un vecteur de score **r** qui représente l'importance de chaque nœud.

#### Formulation Mathématique

Pour chaque nœud v ∈ V, le score PageRank r(v) est calculé itérativement :

```
r^(t+1)(v) = (1 - α) × p(v) + α × Σ_{u→v} [r^(t)(u) × w(u,v) / Σ_{u→w} w(u,w)]
```

Où :
- **α** ∈ [0, 1] : facteur d'amortissement (damping factor) = 0.85 par défaut
- **p(v)** : vecteur de personnalisation (personnalisation vector)
  - p(v_job) = 1
  - p(v) = 0 pour tout autre v
- **t** : numéro d'itération
- **w(u,v)** : poids de l'arête de u vers v

#### En Notation Matricielle

Soit **W** la matrice de transition normalisée où :

```
W[i,j] = w(v_i, v_j) / Σ_k w(v_i, v_k)
```

Alors :

```
r^(t+1) = (1 - α) × p + α × W^T × r^(t)
```

#### Normalisation des Arêtes

Pour obtenir une matrice de transition probabiliste, les poids sortants de chaque nœud sont normalisés :

```
w_norm(u, v) = w(u, v) / Σ_{(u,w)∈E} w(u, w)
```

Ainsi, pour chaque nœud u :

```
Σ_{(u,v)∈E} w_norm(u, v) = 1
```

### Convergence

L'algorithme itère jusqu'à convergence, définie par :

```
||r^(t+1) - r^(t)||₁ < ε
```

Où ε est la tolérance (par défaut 10⁻⁶).

**Théorème de convergence** : Sous les conditions :
1. Le graphe est fortement connexe (ou utilise un vecteur de téléportation)
2. α < 1
3. La matrice W est stochastique

L'algorithme converge vers une distribution stationnaire unique **r*** en temps O(log(1/ε)).

### Score Final Normalisé

Le score final d'un candidat c_j est normalisé par rapport au candidat parfait :

```
score_normalized(c_j) = [r(c_j) / r(v_perfect)] × 100
```

Où :
- **r(c_j)** : score PageRank brut du candidat
- **r(v_perfect)** : score du candidat parfait théorique (qui possède toutes les compétences avec maîtrise maximale)

### Interprétation du Score

Le score normalisé représente un **pourcentage de correspondance idéale** :

- **90-100%** : Correspondance exceptionnelle (très rare)
- **80-89%** : Excellente correspondance
- **70-79%** : Bonne correspondance
- **60-69%** : Correspondance acceptable
- **< 60%** : Correspondance faible

### Complexité Algorithmique

- **Construction du graphe** : O(n_skills × n_candidates)
- **PageRank (par itération)** : O(|E|) = O(n_skills × n_candidates)
- **Convergence** : Typiquement 10-50 itérations
- **Complexité totale** : O(k × n_skills × n_candidates)
  où k est le nombre d'itérations jusqu'à convergence

### Exemple Numérique

Considérons un exemple simplifié :

**Poste** : 2 compétences
- Python : importance = 0.9
- SQL : importance = 0.7

**Candidat A** :
- Python : maîtrise = 0.8, expérience = 5 ans
- SQL : maîtrise = 0.9, expérience = 5 ans

**Calcul** :

1. Boost d'expérience (si requis = 3 ans, préféré = 5 ans) :
   ```
   boost = 1.0 + 0.3 × (5-3)/(5-3) = 1.3
   boost_pondéré = 1.0 + (1.3 - 1.0) × 0.3 = 1.09
   ```

2. Poids des arêtes :
   ```
   w(Python, A) = 0.8 × 1.09 = 0.872
   w(SQL, A) = 0.9 × 1.09 = 0.981
   ```

3. Après normalisation et PageRank, le score final intègre :
   - L'importance des compétences pour le poste
   - La maîtrise du candidat dans chaque compétence
   - Le boost d'expérience
   - La propagation du flux à travers le graphe

## 📦 Installation

### Prérequis

- Python 3.8 ou supérieur
- gestionnaire de paquets pip

### Démarrage Rapide

```bash
# Cloner le dépôt
git clone https://github.com/votrenomdutilisateur/ats-ranker.git
cd ats-ranker

# Créer un environnement virtuel (recommandé)
python -m venv venv
source venv/bin/activate  # Sur Windows : venv\Scripts\activate

# Installer les dépendances
pip install -r requirements.txt
```

### Dépendances Requises

Créer un fichier `requirements.txt` avec :

```
networkx>=3.0
pandas>=1.5.0
streamlit>=1.28.0
plotly>=5.14.0
reportlab>=4.0.0
numpy>=1.24.0
```

## 🚀 Utilisation

### Interface Web Streamlit

```bash
streamlit run src/streamlit_app.py
```

Ensuite, naviguez vers `http://localhost:8501` dans votre navigateur web.

### API Python

```python
from src.graph_ats_ranker import GraphBasedATSRanker
from src.json_loader import ATSDataLoader

# Charger les données
jobs, candidates = ATSDataLoader.load_from_json(
    'data/job_requirements.json',
    'data/candidates.json'
)

# Créer le classeur
ranker = GraphBasedATSRanker(
    experience_weight=0.3,
    experience_mode='both'
)

# Construire le graphe et calculer les classements
ranker.build_graph(jobs[0], candidates)
rankings = ranker.compute_rankings()

# Obtenir une explication détaillée pour un candidat
explanation = ranker.explain_ranking('candidat_001')
print(explanation)
```

### Ligne de Commande (Multi-Postes)

```python
from src.json_loader import rank_from_json

# Classer les candidats pour tous les postes dans le fichier
results = rank_from_json(
    'data/jobs.json',
    'data/candidates.json',
    experience_weight=0.3,
    experience_mode='both'
)

# Results est un dict : { "Titre du Poste": DataFrame, ... }
for job_title, rankings in results.items():
    print(f"\nClassements pour {job_title}:")
    print(rankings.head())
```

## 📄 Formats de Fichiers

### Exigences du Poste

**Poste Unique :**
```json
{
  "title": "Ingénieur Logiciel Senior",
  "min_years_experience": 5,
  "preferred_years_experience": 8,
  "skills": {
    "Python": 0.9,
    "Machine Learning": 0.85,
    "Docker": 0.7,
    "AWS": 0.75,
    "SQL": 0.8
  }
}
```

**Postes Multiples :**
```json
[
  {
    "title": "Ingénieur Logiciel Senior",
    "min_years_experience": 5,
    "skills": { "Python": 0.9, "ML": 0.85 }
  },
  {
    "title": "Data Scientist",
    "min_years_experience": 3,
    "skills": { "Python": 0.85, "Statistiques": 0.9 }
  }
]
```

**Format Alternatif (encapsulé) :**
```json
{
  "jobs": [
    { "title": "Poste 1", "skills": {...} },
    { "title": "Poste 2", "skills": {...} }
  ]
}
```

### Candidats

```json
[
  {
    "id": "Alice Dupont",
    "years_of_experience": 7,
    "skills": {
      "Python": 0.95,
      "Machine Learning": 0.88,
      "Docker": 0.75,
      "AWS": 0.82,
      "SQL": 0.9
    }
  },
  {
    "id": "Bob Martin",
    "years_of_experience": 4,
    "skills": {
      "Python": 0.85,
      "Machine Learning": 0.92,
      "Docker": 0.65
    }
  }
]
```

### Spécifications des Champs

**Exigences du Poste :**
- `title` (chaîne) : Nom du poste
- `min_years_experience` (nombre) : Années minimales requises (défaut : 0)
- `preferred_years_experience` (nombre, optionnel) : Années préférées
- `skills` (objet) : Nom de compétence → importance (0.0-1.0)

**Candidats :**
- `id` (chaîne) : Identifiant unique du candidat
- `years_of_experience` (nombre) : Années d'expérience pertinente
- `skills` (objet) : Nom de compétence → maîtrise (0.0-1.0)

## ⚙️ Options de Configuration

### Modes d'Intégration de l'Expérience

```python
experience_mode='both'     # Recommandé : Utilise boost et arêtes directes
experience_mode='boost'    # Multiplie les poids des compétences par le facteur d'expérience
experience_mode='direct'   # Ajoute des arêtes poste→candidat pour l'expérience
experience_mode='none'     # Ignore l'expérience (compétences uniquement)
```

### Poids de l'Expérience

Contrôle l'influence de l'expérience sur le classement (0.0-1.0) :

- `0.1-0.2` : Influence minimale - les compétences dominent
- `0.3-0.4` : Influence modérée - **recommandé**
- `0.5-0.7` : Influence forte - expérience fortement pondérée
- `0.8-1.0` : Très forte - peut masquer les compétences

### Autres Paramètres

```python
ranker = GraphBasedATSRanker(
    damping=0.85,              # Facteur d'amortissement PageRank (défaut : 0.85)
    tolerance=1e-6,            # Tolérance de convergence (défaut : 1e-6)
    max_iterations=100,        # Itérations PageRank max (défaut : 100)
    normalize_edges=True,      # Normaliser les poids des arêtes (défaut : True)
    experience_weight=0.3,     # Influence de l'expérience (défaut : 0.3)
    experience_mode='both'     # Mode d'expérience (défaut : 'both')
)
```

## 📊 Exemple de Flux de Travail

### 1. Préparer Vos Données

Créer `job.json` :
```json
{
  "title": "Data Scientist",
  "min_years_experience": 3,
  "preferred_years_experience": 5,
  "skills": {
    "Python": 0.9,
    "Statistiques": 0.85,
    "SQL": 0.75,
    "Machine Learning": 0.8
  }
}
```

Créer `candidates.json` :
```json
[
  {
    "id": "Candidat_A",
    "years_of_experience": 5,
    "skills": {
      "Python": 0.9,
      "Statistiques": 0.85,
      "SQL": 0.8,
      "Machine Learning": 0.75
    }
  },
  {
    "id": "Candidat_B",
    "years_of_experience": 2,
    "skills": {
      "Python": 0.95,
      "Statistiques": 0.7,
      "Machine Learning": 0.9
    }
  }
]
```

### 2. Exécuter le Classeur

```python
from src.json_loader import rank_from_json

results = rank_from_json('job.json', 'candidates.json')
rankings = results['Data Scientist']
print(rankings)
```

**Sortie :**
```
  candidate_id     score  normalized_score  years_experience  rank
0  Candidat_A   0.045123         89.234516               5.0     1
1  Candidat_B   0.038456         76.123892               2.0     2
```

### 3. Obtenir des Explications Détaillées

```python
from src.graph_ats_ranker import GraphBasedATSRanker
from src.json_loader import ATSDataLoader

jobs, candidates = ATSDataLoader.load_from_json('job.json', 'candidates.json')

ranker = GraphBasedATSRanker()
ranker.build_graph(jobs[0], candidates)
ranker.compute_rankings()

explanation = ranker.explain_ranking('Candidat_A')
print(f"Rang : {explanation['rank']}")
print(f"Score : {explanation['normalized_score']:.2f}")
print(f"Statut d'expérience : {explanation['experience_status']}")
print(f"Couverture de compétences : {explanation['skill_coverage']:.1%}")
```

## 🏗️ Architecture

### Structure du Projet

```
ats-ranker/
├── src/
│   ├── graph_ats_ranker.py    # Algorithme de classement principal
│   ├── json_loader.py          # Chargement et validation des données
│   └── streamlit_app.py        # Interface web
├── requirements.txt
└── README.md
```

### Composants Principaux

**GraphBasedATSRanker** (`graph_ats_ranker.py`)
- Construction de graphe avec NetworkX
- Calcul PageRank
- Logique d'intégration de l'expérience
- Génération d'explications de classement

**ATSDataLoader** (`json_loader.py`)
- Analyse JSON multi-format
- Validation des données
- Support pour fichiers mono et multi-postes

**Application Streamlit** (`streamlit_app.py`)
- Interface web interactive
- Génération de visualisations
- Exports de rapports (CSV, PDF)

## 📚 Référence API

### GraphBasedATSRanker

#### Constructeur
```python
ranker = GraphBasedATSRanker(
    damping=0.85,
    tolerance=1e-6,
    max_iterations=100,
    normalize_edges=True,
    experience_weight=0.3,
    experience_mode='both'
)
```

#### Méthodes

**`build_graph(job_requirements, candidates)`**
Construit le graphe de classement.
- **Paramètres :**
  - `job_requirements` (dict) : Spécification du poste
  - `candidates` (list) : Liste de dictionnaires de candidats
- **Retourne :** NetworkX DiGraph

**`compute_rankings()`**
Exécute PageRank et génère les classements.
- **Retourne :** pandas DataFrame avec colonnes :
  - `candidate_id` : Identifiant du candidat
  - `score` : Score PageRank brut
  - `normalized_score` : Score en % du candidat parfait
  - `years_experience` : Années d'expérience
  - `rank` : Position de classement (1 = meilleur)

**`explain_ranking(candidate_id, top_k_skills=6)`**
Génère une explication détaillée pour le classement d'un candidat.
- **Paramètres :**
  - `candidate_id` (str) : Candidat à expliquer
  - `top_k_skills` (int) : Nombre de compétences principales à inclure
- **Retourne :** Dictionnaire avec :
  - `rank`, `score`, `normalized_score`
  - `top_skills` : Liste de compétences correspondantes avec contributions
  - `missing_skills` : Liste de lacunes
  - `skill_coverage` : Pourcentage de compétences requises possédées
  - `experience_status` : Évaluation de l'expérience

**`get_graph_stats()`**
Retourne les statistiques du graphe et la configuration.
- **Retourne :** Dictionnaire avec métriques du graphe

### ATSDataLoader

**`load_from_json(job_file, candidates_file)`**
- **Retourne :** Tuple de (liste de postes, liste de candidats)

**`validate_data(job_requirements, candidates)`**
- **Retourne :** Tuple de (is_valid: bool, errors: list)

**`save_to_json(job_requirements, candidates, job_output_file, candidates_output_file)`**
- Sauvegarde les données dans des fichiers JSON

### Fonctions Auxiliaires

**`rank_from_json(job_file, candidates_file, experience_weight=0.3, experience_mode='both')`**
- Classement en une étape pour fichiers multi-postes
- **Retourne :** Dict mappant titres de postes aux DataFrames de classement

## 🧪 Tests

Exécuter les tests avec pytest :

```bash
pytest tests/
```

Exemple de test :

```python
def test_basic_ranking():
    job = {
        'title': 'Poste Test',
        'skills': {'Python': 0.9, 'SQL': 0.7}
    }
    candidates = [
        {'id': 'A', 'years_of_experience': 5, 'skills': {'Python': 0.9, 'SQL': 0.8}},
        {'id': 'B', 'years_of_experience': 2, 'skills': {'Python': 0.7}}
    ]
    
    ranker = GraphBasedATSRanker()
    ranker.build_graph(job, candidates)
    rankings = ranker.compute_rankings()
    
    assert rankings.iloc[0]['candidate_id'] == 'A'
    assert len(rankings) == 2
```

## 🤝 Contribuer

Les contributions sont les bienvenues ! N'hésitez pas à soumettre une Pull Request.

### Configuration de Développement

```bash
# Forker et cloner le dépôt
git clone https://github.com/votrenomdutilisateur/ats-ranker.git
cd ats-ranker

# Créer une branche de développement
git checkout -b feature/votre-nom-de-fonctionnalite

# Faire vos modifications et tester
pytest tests/

# Soumettre une pull request
```

### Directives

- Suivre les directives de style PEP 8
- Ajouter des tests pour les nouvelles fonctionnalités
- Mettre à jour la documentation si nécessaire
- Garder les commits atomiques et bien décrits

## 📝 Licence

Ce projet est sous licence MIT - voir le fichier LICENSE pour plus de détails.

## 🙏 Remerciements

- L'équipe NetworkX pour la bibliothèque d'algorithmes de graphes
- Streamlit pour le framework web extraordinaire
- L'algorithme PageRank originalement développé par Larry Page et Sergey Brin

## 📧 Support

Pour questions, problèmes ou suggestions :
- Ouvrir un issue sur GitHub
- Contact : laraisse66@gmail.com

## 🗺️ Feuille de Route

- [ ] Ajouter le support pour plus de formats de fichiers (Excel, CSV)
- [ ] Implémenter des synonymes de compétences et correspondance floue
- [ ] Ajouter le traitement par lots pour de grands pools de candidats
- [ ] Créer un endpoint API REST
- [ ] Ajouter le support pour l'expérience pondérée par domaine
- [ ] Implémenter des recommandations par filtrage collaboratif
- [ ] Ajouter la visualisation de la structure du graphe

---

**Fait avec ❤️ pour de meilleures décisions d'embauche**