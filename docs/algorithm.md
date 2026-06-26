# Algoritmos

Comparativa entre los algoritmos de la lista objetivo y los configs disponibles en el proyecto.

| Algoritmo | Disponible | Nombre en el proyecto |
|---|---|---|
| MLP | ✅ | `MLP` |
| LightGBM+GFs | ❌ | No implementado (requiere framework externo) |
| XGBoost+GFs | ❌ | No implementado (requiere framework externo) |
| GatedGCN | ✅ | `GatedGCN` |
| GAT | ❌ | No implementado (hay `GATE`, que es distinto) |
| GIN | ✅ | `GINE` |
| GIN+RMP | ✅ | `GINE+RMP` |
| GIN+Ports | ✅ | `GINE+ports` |
| GIN+Ego ID | ✅ | `GINE+Ego` |
| GIN+EU | ✅ | `GINE+EU` |
| PNA | ✅ | `PNA` |
| PNA+EU | ✅ | `PNA+EU` |
| Multi-GIN | ✅ | `Multi-GINE` |
| Multi-GIN+EU | ✅ | `Multi-GINE+EU` |
| Multi-PNA | ✅ | `Multi-PNA` |
| Multi-PNA+EU | ✅ | `Multi-PNA+EU` |
| FraudGT | ✅ | `SparseNodeGT` |
| FraudGT+RMP | ✅ | `SparseNodeGT+RMP` |
| FraudGT+Ports | ✅ | `SparseNodeGT+ports` |
| FraudGT+Ego ID | ✅ | `SparseNodeGT+Ego` |
| FraudGT+Ports+Ego ID (PE-FraudGT) | ✅ | `SparseNodeGT+ports+Ego` |
| FraudGT+RMP+Ports+Ego ID (Multi-FraudGT) | ✅ | `Multi-SparseNodeGT` |

Los configs se encuentran en `configs/AML-Small-HI/` con el prefijo `AML-Small-HI-`.
