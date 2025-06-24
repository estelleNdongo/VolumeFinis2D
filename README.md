# Solveur d'équations de Poisson par volumes finis 2D

## Structure du projet

```
finite_volume_solver/
├── README.md                 # Ce fichier
├── main.py                   # Script principal
├── config/
│   ├── __init__.py
│   └── config.py            # Configuration et paramètres
├── src/
│   ├── __init__.py
│   ├── mesh/
│   │   ├── __init__.py
│   │   └── mesh_generator.py           # Génération de maillages
│   ├── boundary/
│   │   ├── __init__.py
│   │   └── boundary_conditions.py      # Conditions aux limites
│   ├── solver/
│   │   ├── __init__.py
│   │   └── finite_volume_solver.py       # Solveur volumes finis
│   ├── functions/
│   │   ├── __init__.py
│   │   └── source_functions.py           # Fonctions sources et solutions exactes
│   └── analysis/
│       ├── __init__.py
│       ├── error_analysis.py               # Analyse d'erreur et convergence
│       └── visualization.py                # Visualisation des résultats
└── tests/
    ├── __init__.py
    ├── test_mesh.py
    ├── test_solver.py
    └── test_convergence.py
```

## Utilisation

```python
python main.py --nx 20 --ny 20 --function quadratic --bc_type dirichlet
```

## Fonctionnalités

- **Maillages** : rectangulaires uniformes, adaptatifs
- **Fonctions sources** : polynomiales, trigonométriques, personnalisées
- **Conditions aux limites** : Dirichlet, Neumann, mixtes
- **Analyse** : convergence, erreurs L2/L∞, visualisation
- **Extensibilité** : architecture modulaire pour nouvelles EDP