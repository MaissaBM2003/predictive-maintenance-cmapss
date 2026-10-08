# Predictive Maintenance — NASA C-MAPSS

Prédiction de la durée de vie restante (RUL) de turboréacteurs et détection d'anomalies,
avec un pipeline ML et MLOps complet.

## Installation
poetry install

## Structure
- data/ : données brutes et prétraitées
- notebooks/ : exploration
- src/ : code source
- tests/ : tests
- configs/ : configuration




## Résultats sur FD001

| Modèle | Val RMSE | Test RMSE | Test MAE | Score NASA |
|---|---|---|---|---|
| Régression linéaire | 15,06 | 15,71 | 12,73 | 441,7 |
| Random Forest | 12,96 | 13,14 | 9,20 | 302,5 |
| **LightGBM** | 12,63 | **13,13** | **9,76** | **278,1** |
| GRU | **11,77** | 13,75 | 10,01 | 336,7 |
| LSTM | 12,10 | 14,29 | 10,49 | 385,0 |
| Transformer | 12,10 | 14,63 | 10,70 | 415,0 |
| CNN 1D | 13,88 | 15,84 | 11,74 | 507,7 |