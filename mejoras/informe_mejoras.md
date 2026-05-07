# Informe de mejoras - FraudGT (AML Small-HI, CPU)

## 1) Contexto y ejecucion
- Dataset: AML Small-HI
- Archivo esperado: data/AML/HI-Small_Trans.csv
- Config base: configs/AML-Small-HI/AML-Small-HI-GINE.yaml
- Dispositivo: CPU
- Cambios para ejecucion corta:
  - train.iter_per_epoch: 64
  - val.iter_per_epoch: 32
  - optim.max_epoch: 40
  - num_workers: 0

## 2) Resultados actuales (run 42)
Fuente: results/AML-Small-HI/AML-Small-HI-GINE/42/{train,val,test}/stats.json

Mejor validacion (segun f1):
- Epoch 35
- Val: f1 0.24719, precision 0.47826, recall 0.16667, auc 0.95743

Mejor test (segun f1):
- Epoch 39
- Test: f1 0.31016, precision 0.47541, recall 0.23016, auc 0.94710

Observacion:
- Accuracy y micro-f1 son ~0.998-0.999, pero el recall es bajo.
  Esto sugiere un fuerte desbalance de clases; f1/recall son mas informativos.

## 3) Diagnostico (hallazgos)
1) Desbalance fuerte: el modelo puede predecir casi todo como negativo y aun asi tener accuracy alta.
2) Entrenamiento corto: el f1 aun mejora hacia las ultimas epocas, asi que mas epochs podrian ayudar.
3) Umbral fijo 0.5: puede ser demasiado alto para la clase positiva.
4) CPU-only: limita el tamano de busqueda de hiperparametros.

## 4) Propuestas de mejora (prioridad alta a baja)
Alta prioridad (impacto en f1/recall):
- Ajustar el peso de clase: loss_fun_weight [1, 6] -> [1, 10] o [1, 20].
- Ajustar el umbral de decision: probar 0.1 a 0.5 y escoger el mejor f1 en val.
- Aumentar entrenamiento: iter_per_epoch 128 y max_epoch 60 (o early stopping por f1).

Prioridad media:
- Probar variantes con features: GINE+ports o GINE+RMP (configs existentes).
- Probar dropout moderado (0.1-0.3) para mejorar generalizacion.
- Ajustar batch_size (1024 o 4096) si el tiempo lo permite.

Prioridad baja / exploratoria:
- Cambiar modelo a SparseNodeGT o SparseEdgeGT (mas caro en CPU).
- Probar posenc Hetero_Node2Vec si se permite preprocesamiento extra.

## 5) Experimentos concretos sugeridos
1) Reponderacion de clases:
   - loss_fun_weight: [1, 10] y [1, 20].
2) Umbral de decision:
   - barrido en val: 0.1, 0.2, 0.3, 0.4, 0.5.
3) Entrenamiento mas largo:
   - iter_per_epoch: 128
   - max_epoch: 60
4) Variante de features:
   - GINE+ports (config AML-Small-HI-GINE+ports.yaml)

## 6) Salidas y ubicaciones
- Logs: results/AML-Small-HI/AML-Small-HI-GINE/42/logging.log
- Stats: results/AML-Small-HI/AML-Small-HI-GINE/42/{train,val,test}/stats.json
- Config final: results/AML-Small-HI/AML-Small-HI-GINE/config.yaml

## 7) Proximo paso recomendado
Priorizar mejorar recall/f1 con:
- loss_fun_weight mas alto y umbral mas bajo,
- y entrenar un poco mas (sin llegar a 7-8 horas).
