# Protocolo experimental y adaptación de baselines

## Alcance y jerarquía de fuentes

Este informe reconstruye la configuración experimental dando prioridad a los
cuadernos de `training_notebooks/`:

1. `training_notebooks/train_main_model/`: experimentos de OutGrid.
2. `training_notebooks/train_comparision_models/`: baselines de CIFAR-100.
3. `src/`: se consulta únicamente para interpretar las funciones llamadas por
   los cuadernos (optimizador, scheduler, particiones, Mixup/CutMix y guardado
   del mejor checkpoint).

Los archivos YAML no se usan como evidencia de los experimentos reportados.

## Resumen ejecutivo

- Los baselines y OutGrid-7M/14M sobre CIFAR-100 a 32 px se entrenaron bajo el
  mismo protocolo general: 100 épocas, batch 64, AdamW, LR `5e-4`, weight decay
  `0.05`, warm-up lineal del 5% de los pasos y decaimiento cosenoidal hasta
  `1e-6`.
- CIFAR-100 se divide en 45 000 imágenes de entrenamiento, 5 000 de validación
  y 10 000 de test. La partición train/val se genera con seed 7.
- Los experimentos de SVHN, Tiny ImageNet-200 y CIFAR-100 a 64 px usan 50
  épocas, batch 64 y el mismo optimizador/scheduler, pero desactivan Mixup y
  configuran únicamente CutMix con probabilidad 0.5.
- No hay evaluación multiseed ni reporte de media/desviación: cada resultado
  proviene de una sola corrida.
- El checkpoint `best` se selecciona por accuracy top-1 de validación, con una
  mejora mínima de 0.05 puntos porcentuales y paciencia de 6 épocas.
- Los metadatos identifican GPU NVIDIA Tesla T4 para los baselines. Los
  cuadernos grandes de OutGrid indican A100 en Colab, aunque el cuaderno
  CIFAR-100 14M contiene metadatos contradictorios A100/T4.

## Datasets, resolución y particiones

| Experimento         |           Dataset | Resolución | Partición efectiva                                                               | Fuente principal                                               |
| ------------------- | ----------------: | ----------: | --------------------------------------------------------------------------------- | -------------------------------------------------------------- |
| OutGrid 7M          |         CIFAR-100 |      32×32 | 45 000 train / 5 000 val / 10 000 test                                            | `OutGridViT_Cifrar32_7M.ipynb`, celdas 1 y 7                 |
| OutGrid 14M         |         CIFAR-100 |      32×32 | 45 000 train / 5 000 val / 10 000 test                                            | `OutGridViT_ModelA_Cifrar32_14M.ipynb`, celdas 1 y 5         |
| OutGrid 14M         |         CIFAR-100 |      64×64 | 45 000 train / 5 000 val / 10 000 test; las imágenes se redimensionan de 32 a 64 | `OutGridViT_Models_Cifrar64_14M.ipynb`, celdas 1 y 5         |
| OutGrid 14M         |              SVHN |      32×32 | 65 932 train / 7 325 val / 26 032 test                                            | `OutGridViT_ModelA_SVHN_14M.ipynb`, celdas 1 y 6             |
| OutGrid 22M         | Tiny ImageNet-200 |      64×64 | 90 000 train / 10 000 val / 10 000 test efectivo                                  | `OutGridViT_ModelA_Tiny_Imagenet200_22M.ipynb`, celdas 1 y 8 |
| Todos los baselines |         CIFAR-100 |      32×32 | 45 000 train / 5 000 val / 10 000 test                                            | celda 1 de cada cuaderno de`train_comparision_models/`       |

### Cómo se construyen las particiones

- Todos los cuadernos pasan `val_split=0.1`.
- En CIFAR-100 y SVHN, el 10% se extrae del split oficial de entrenamiento
  mediante `random_split`, con seed 7. El test es el split oficial.
- En Tiny ImageNet-200, el 10% se extrae de las 100 000 imágenes de
  entrenamiento. Cuando se hace esta división, el split oficial
  `validation` se devuelve como `test_loader`; de ahí los 90k/10k/10k.
- La validación creada por `random_split` comparte el objeto dataset y,
  por tanto, el transform de entrenamiento. Esto significa que la validación
  interna recibe las transformaciones aleatorias de train (crop, flip,
  RandAugment y Random Erasing), no el transform determinista de test.
- El pipeline de train que heredan los cuadernos usa RandomCrop,
  RandomHorizontalFlip, RandAugment (`num_ops=2`, magnitud 7), normalización y
  Random Erasing (`p=0.25`). Para CIFAR-100 a 64 px se añade Resize bicúbico.

## Configuración de entrenamiento de OutGrid

| Cuaderno / corrida        | Batch | Épocas | Seed de entrenamiento | LR / WD             | Warm-up / mínimo | Mix prob. | Mixup α | CutMix α | Label smoothing configurado | AMP  |
| ------------------------- | ----: | ------: | --------------------: | ------------------- | ----------------- | --------: | -------: | --------: | --------------------------: | ---- |
| CIFAR-100 32, 7M          |    64 |     100 |                     7 | `5e-4` / `0.05` | 5% /`1e-6`      |       0.5 |      0.8 |       1.0 |                         0.1 | FP16 |
| CIFAR-100 32, 14M         |    64 |     100 |                    77 | `5e-4` / `0.05` | 5% /`1e-6`      |       0.5 |      0.8 |       1.0 |                         0.1 | FP16 |
| CIFAR-100 64, 14M         |    64 |      50 |                     7 | `5e-4` / `0.05` | 5% /`1e-6`      |       0.5 |      0.0 |       1.0 |                         0.0 | FP16 |
| SVHN 32, 14M              |    64 |      50 |                     7 | `5e-4` / `0.05` | 5% /`1e-6`      |       0.5 |      0.0 |       1.0 |                         0.0 | FP16 |
| Tiny ImageNet-200 64, 22M |    64 |      50 |                     7 | `5e-4` / `0.05` | 5% /`1e-6`      |       0.5 |      0.0 |       1.0 |                         0.0 | FP16 |

Todas estas corridas también usan gradient clipping con norma 1.0,
`channels_last=True` y AMP en CUDA.

### Optimizador, warm-up y scheduler

Los cuadernos llaman a `train_model`, que construye:

- AdamW con `betas=(0.9, 0.999)` y `eps=1e-8`.
- Weight decay de 0.05 para pesos ordinarios.
- Weight decay 0 para biases, normalizaciones, embeddings posicionales y
  class tokens.
- Scheduler por paso, no por época:
  - `total_steps = epochs × len(train_loader)`;
  - warm-up lineal durante el 5% de los pasos;
  - después, cosine decay hasta `min_lr=1e-6`.

Los logs incluidos en las salidas de los cuadernos confirman:

- CIFAR-100, 100 épocas: 70 400 pasos, 3 520 de warm-up.
- CIFAR-100 a 64 px, 50 épocas: 35 200 pasos, 1 760 de warm-up.
- SVHN, 50 épocas: 51 550 pasos, 2 577 de warm-up.
- Tiny ImageNet-200, 50 épocas: 70 300 pasos, 3 515 de warm-up.

## Mixup, CutMix y label smoothing: configuración y efecto real

### CIFAR-100 a 32 px y baselines

Los cuadernos configuran `mix_prob=0.5`, `mixup_alpha=0.8`,
`cutmix_alpha=1.0` y `label_smoothing=0.1`.

Cuando se activa la mezcla para un batch, el código elige Mixup o CutMix con
probabilidad 0.5 cada uno. En términos aproximados por batch:

- 50% sin mezcla;
- 25% Mixup;
- 25% CutMix.

### SVHN, Tiny ImageNet-200 y CIFAR-100 a 64 px

Configuran `mix_prob=0.5`, `mixup_alpha=0.0`, `cutmix_alpha=1.0` y
`label_smoothing=0.0`: aproximadamente 50% de batches con CutMix y 50% sin
mezcla.

### Matiz de implementación

En la versión con la que se ejecutaron los cuadernos, si `mixup_alpha > 0`
**o** `cutmix_alpha > 0`, la pérdida usada era soft-target cross entropy y el
`label_smoothing=0.1` quedaba sin efecto. El pipeline actual de `src/` ya
corrige este punto: suaviza los targets blandos sin descartar la distribución
producida por Mixup/CutMix. Esta diferencia debe registrarse si se comparan
resultados históricos con nuevas corridas controladas por YAML.

## Seeds y reproducibilidad

### OutGrid

- Cada cuaderno ejecuta una sola seed de entrenamiento; no hay bucle sobre
  varias seeds.
- Se fijan `torch`, `torch.cuda`, Python `random` y NumPy.
- Seed de entrenamiento:
  - 7 en OutGrid 7M, SVHN, Tiny ImageNet-200 y CIFAR-100 a 64 px;
  - 77 en OutGrid 14M sobre CIFAR-100 a 32 px.
- La partición/dataloader de CIFAR-100 14M sigue usando seed 7 aunque el
  entrenamiento usa seed 77.
- reproducibilidad de la inicialización de pesos. El cuaderno de CIFAR-100 a
  64 px sí fija la seed antes de construir el modelo.
- `torch.backends.cudnn.benchmark=True`; por ello no se solicita ejecución
  estrictamente determinista.

### Baselines

- Los nueve cuadernos usan seed 7 para la partición y el generador del
  dataloader.
- No fijan explícitamente una seed global antes de crear/inicializar el modelo.
- También corresponden a una sola corrida por arquitectura.

## Criterio para el mejor resultado

Los cuadernos no pasan argumentos de early stopping, por lo que heredan de
`train_model`:

- métrica: `val top-1`;
- modo: maximización;
- mejora mínima: más de 0.05 puntos porcentuales;
- paciencia: 6 épocas;
- early stopping habilitado.

El checkpoint `best_*.pt` se sobrescribe cuando `val top-1` supera al mejor
valor previo por más de 0.05. También se guarda un checkpoint `last` en cada
época.

## Hardware

La siguiente información procede de los metadatos embebidos en los
cuadernos, no de los YAML:

| Grupo / cuaderno              | Hardware indicado                                                                              |
| ----------------------------- | ---------------------------------------------------------------------------------------------- |
| Los nueve baselines           | NVIDIA Tesla T4; la mayoría registra tanto Colab`gpuType: T4` como Kaggle `nvidiaTeslaT4` |
| OutGrid CIFAR-100 7M          | NVIDIA Tesla T4                                                                                |
| OutGrid CIFAR-100 64 px       | NVIDIA Tesla T4                                                                                |
| OutGrid SVHN 14M              | NVIDIA A100 en metadatos de Colab                                                              |
| OutGrid Tiny ImageNet-200 22M | NVIDIA A100 en metadatos de Colab                                                              |
| OutGrid CIFAR-100 14M         | Ambiguo: metadatos de Colab indican A100 y los de Kaggle indican Tesla T4                      |

Todos entrenan con CUDA, AMP y formato `channels_last`. La precisión es FP16
salvo DeiT-Nano y Swin, cuyos cuadernos solicitan BF16. En el cuaderno de
CIFAR-100 64 px aparece además el comentario `#10.5 GB`, pero no identifica
por sí solo el modelo de GPU.

## Baselines comparados con OutGrid en CIFAR-100

Todos usan imágenes 32×32, batch 64, 100 épocas y el mismo protocolo general
de optimización/augmentación de CIFAR-100 descrito arriba.

| Nombre usado en el proyecto | Modelo realmente instanciado     | Parámetros mostrados | Adaptación para 32×32                                                                                           |                                Top-1 de test impreso |
| --------------------------- | -------------------------------- | --------------------: | ----------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------: |
| DeiT-Nano                   | `deit_tiny_patch16_224`        |                 5.38M | Se construye con`img_size=32` y `patch_size=4`, en lugar del patch 16 original                                |                                                63.77 |
| DeiT-Tiny                   | `deit_small_patch16_224`       |                21.38M | Se construye con`img_size=32` y `patch_size=4`                                                                |                                                59.00 |
| SwinViT                     | `swin_tiny_patch4_window7_224` |                27.57M | `img_size=32`, ventana 4; patch embedding reemplazado por conv 2×2, stride 2, y `patch_size=(2,2)`           |                                                59.89 |
| ConvNeXt-Tiny               | `convnext_tiny`                |                27.90M | Stem reemplazado por conv 2×2, stride 2, 3→96 canales, sin padding                                              |                                                72.60 |
| EfficientNetV2-S            | `efficientnetv2_s`             |                20.31M | `conv_stem` reemplazado por conv 3×3, stride 1, padding 1; conserva el número de canales de salida            | 64.62 (64.66 en otra evaluación del mismo cuaderno) |
| MaxViT-Nano                 | `maxvit_tiny_tf_224` custom    |                17.38M | `embed_dim=[64,96,192,384]`; stem conv1 y conv2 pasan a 3×3, stride 1, padding 1; norm1 se ajusta a 64 canales |                                                75.41 |
| MaxViT-Tiny                 | `maxvit_tiny_tf_224`           |                30.43M | Stem conv1 y conv2 reemplazados por 3×3, stride 1, padding 1, conservando canales                                |                                                75.90 |
| ResNet-18                   | `resnet18`                     |                11.22M | Conv inicial 7×7/stride 2 → 3×3/stride 1/padding 1; max-pool inicial → identidad                              |                                                73.25 |
| ResNet-50                   | `resnet50`                     |                23.71M | Conv inicial 7×7/stride 2 → 3×3/stride 1/padding 1; max-pool inicial → identidad                              |                                                77.42 |

Como referencia directa en los cuadernos, OutGrid obtiene 79.72 top-1 con
7M y 80.85 top-1 con 14M sobre CIFAR-100 a 32×32. Estos valores deben
interpretarse con la advertencia anterior: las celdas de evaluación no
recargan explícitamente el checkpoint seleccionado como `best`.

### Qué se buscó con las modificaciones de stem

Las arquitecturas originales están diseñadas principalmente para ImageNet a
224×224 y reducen la resolución demasiado pronto para entradas 32×32. Las
modificaciones disminuyen el stride inicial, eliminan el max-pooling temprano
o reducen el tamaño del patch. Con ello se conserva una rejilla espacial más
densa en las primeras etapas y la comparación con OutGrid resulta más
apropiada para CIFAR-100.

### Inconsistencias nominales que conviene corregir al escribir el artículo

- `Deit_nano.ipynb` no instancia una variante oficial “nano”; usa
  `deit_tiny_patch16_224` con patch 4.
- `Deit_tiny.ipynb` instancia realmente `deit_small_patch16_224`.
- MaxViT-Nano es una reducción personalizada de MaxViT-Tiny, no un identificador
  independiente de `timm`.

## Resultados de OutGrid impresos por los cuadernos

Estos valores son útiles para identificar qué corrida está documentada, pero
no sustituyen la advertencia sobre el checkpoint final frente al `best`:

| Dataset / resolución    | Variante    | Top-1 impreso |
| ------------------------ | ----------- | ------------: |
| CIFAR-100 32×32         | OutGrid 7M  |         79.72 |
| CIFAR-100 32×32         | OutGrid 14M |         80.85 |
| CIFAR-100 64×64         | OutGrid 14M |         81.15 |
| SVHN 32×32              | OutGrid 14M |        97.576 |
| Tiny ImageNet-200 64×64 | OutGrid 22M |         69.84 |

El cuaderno de Tiny ImageNet también imprime 71.09 sobre un subconjunto
denominado `clean_182`; no debe confundirse con el resultado estándar de
69.84 sobre el loader de test/validación oficial.
