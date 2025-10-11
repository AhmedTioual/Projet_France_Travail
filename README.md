Projet_Emploi_Salaire/
│
├── 0_data/                      # Données brutes et préparées
│   ├── raw/                     # Données brutes téléchargées via API France Travail
│   │   └── offres_raw.json
│   ├── processed/               # Données nettoyées, prêtes pour analyse
│   │   └── offres_clean.csv
│   └── dictionnaires/           # Codes métiers, régions, etc.
│       └── mapping_region.csv
│
├── 1_preprocessing/             # Scripts de nettoyage et préparation
│   ├── 01_extraction_API.R      # Récupération données depuis API
│   ├── 02_nettoyage.R           # Nettoyage : NA, doublons, variables utiles
│   ├── 03_feature_engineering.R # Encodage variables, discrétisation salaire, etc.
│   └── README.md                # Explication du pipeline de préparation
│
├── 2_supervised_regression/     # Partie Classification supervisée (régression)
│   ├── regression_lm.R          # Régression linéaire multiple
│   ├── regression_rpart.R       # Arbre de régression
│   ├── regression_randomforest.R# Random Forest
│   └── results/                 # Résultats (graphiques, tableaux comparatifs)
│       └── comparaison_modeles.png
│
├── 3_unsupervised_clustering/   # Partie Classification non supervisée
│   ├── clustering_kmeans.R
│   ├── clustering_CAH.R
│   └── results/
│       └── clusters_profil_emplois.png
│
├── 4_association_rules/         # Partie Règles d’association
│   ├── rules_apriori.R
│   ├── rules_visualisation.R
│   └── results/
│       └── regles_salaire.html
│
├── 5_analysis_reporting/        # Analyse finale et rapport
│   ├── synthese_resultats.R     # Comparaison régression / clustering / règles
│   ├── visualisations.R         # Graphiques globaux
│   ├── rapport.Rmd              # Rapport R Markdown (peut générer PDF/HTML)
│   └── presentation.pptx        # Slides de soutenance
│
├── utils/                       # Fonctions réutilisables
│   └── helpers.R                # Ex: fonctions de nettoyage, visualisation
│
└── README.md                    # Plan global du projet + instructions
