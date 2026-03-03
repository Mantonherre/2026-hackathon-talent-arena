# Modificaciones realizadas al repositorio

Este documento describe todos los cambios implementados sobre el repositorio base del hackathon ALIA Talent Arena 2026, con el objetivo de maximizar la puntuación en los criterios de **Accuracy**, **Robustez** y **Contribución al Dataset**.

---

## Resumen ejecutivo

Se han realizado mejoras en 4 áreas principales:
1. **Prompt Engineering**: rúbricas por categoría + separación historial/pregunta
2. **Calidad de datos**: filtrado de muestras inválidas + razonamiento enriquecido en SFT
3. **Augmentación**: triplicación del dataset con variantes ruidosas para robustez
4. **Configuración LoRA**: hiperparámetros optimizados para Prometheus-7B

---

## 1. `src/prompts.py` — Reescritura completa

### Qué se cambió
- Añadido `CATEGORY_RUBRICS`: diccionario con rúbricas específicas para cada categoría de seguridad.
- Añadida función `get_category_rubric(category_name)` con fallback a rúbrica genérica.
- `ABSOLUTE_PROMPT` reescrito con:
  - Separación explícita de `{history_str}` (historial previo) y `{question}` (última pregunta del usuario)
  - Instrucciones paso a paso para el juez
  - Criterios PASSED/FAILED con ejemplos concretos
  - Variable `{category_rubric}` inyectada automáticamente desde la categoría del sample
- `ABS_SYSTEM_PROMPT` mejorado con mayor autoridad y especificidad

### Por qué mejora el resultado
El prompt original usaba una rúbrica genérica igual para todas las categorías. Un jailbreak de privacidad necesita criterios distintos a uno de odio. Con rúbricas por categoría, el juez LLM tiene contexto específico para cada tipo de ataque, lo que incrementa la precisión de las predicciones.

La separación de `{history_str}` y `{question}` permite que el pipeline de robustez aplique el ruido **solo** a la última pregunta del usuario, manteniendo el historial limpio. Esto simula correctamente cómo llegan los prompts adversariales en la realidad.

---

## 2. `src/data_utils.py` — Ampliación significativa

### Funciones nuevas

#### `filter_high_quality(df)`
Filtra el dataset eliminando muestras con `val_context_bool=False` (la conversación no era pertinente al reto evaluado). Elimina 11 de 80 muestras (13.75% de ruido de etiquetado).

#### `generate_reasoning_template(sample)`
Genera razonamiento estructurado para SFT. Prioriza `val_goal_reasoning` del dataset si tiene más de 30 caracteres (razonamiento validado por humanos). Como fallback, genera un texto coherente basado en categoría, veredicto y challenge.

#### `augment_with_noise(df, seed=42)`
Triplica el dataset con variantes ruidosas usando `promptnoises`:
- **Typos**: errores de teclado QWERTY, omisión de letras
- **Gramatical**: seseo/ceceo, confusión b/v, eliminación de h inicial

### Cambios en funciones existentes

- `load_data()`: encoding UTF-8 explícito (corrección de bug en Windows)
- `prepare_dataset()`: extrae `val_goal_reasoning`, `val_context_bool`, `val_stop_reason` del campo `validation`; extrae `history_str` separado de la última pregunta
- `get_last_valid_turn()`: ahora devuelve `history_str` (historial sin la última pregunta) además de `conversation` (historial completo)
- `format_instruction()`: inyecta `{category_rubric}` automáticamente llamando a `get_category_rubric`
- `prepare_sft_binary_text()`: usa `generate_reasoning_template()` en vez del placeholder genérico `"The response is evaluated based on the provided rubric."`

### Por qué mejora el resultado
El filtrado de calidad elimina ruido de etiquetado que degrada el entrenamiento. El razonamiento enriquecido en SFT enseña al modelo *por qué* una respuesta es segura o insegura, no solo el veredicto binario. La augmentación con ruido mejora directamente la métrica de robustez.

---

## 3. `notebooks/02_fine-tuning.ipynb` — Mejoras de entrenamiento

### Cambios principales

| Aspecto | Original | Mejorado |
|---------|----------|---------|
| Dataset | `dataset_sample.json` (10 muestras) | `dataset_train_ds.json` (80 muestras) |
| Filtrado | Ninguno | `filter_high_quality()` → 69 muestras |
| Augmentación | No | `augment_with_noise()` → ~207 muestras en train |
| Split | Aleatorio simple | Variantes ruidosas solo en train (sin contaminación) |
| Razonamiento SFT | Placeholder genérico | `generate_reasoning_template()` |

### Configuración LoRA mejorada

| Parámetro | Original | Mejorado | Justificación |
|-----------|----------|---------|---------------|
| `r` (rango) | 16 | **64** | Mayor capacidad para matices semánticos |
| `lora_alpha` | 32 | **128** | Ratio 2:1 respecto al rango (óptimo documentado) |
| `target_modules` | q, v | **q, k, v, o, gate, up, down** | Cubre 100% de los parámetros entrenables de Mistral |
| `lora_dropout` | 0.05 | **0.05** | Regularización estándar |
| `num_train_epochs` | 1 | **3** | Suficiente para dataset pequeño sin overfitting |
| `learning_rate` | default | **2e-4** | Óptimo documentado para LoRA en modelos 7B |
| `lr_scheduler_type` | — | **cosine** | Decaimiento suave sin oscilaciones |
| `warmup_ratio` | — | **0.05** | 5% de pasos de calentamiento |
| `weight_decay` | 0 | **0.01** | Regularización L2 contra overfitting |
| `fp16` | — | **True** | Entrenamiento en media precisión (A100) |
| `max_seq_length` | 1024 | **2048** | Soporta conversaciones multi-turno largas |

### Análisis por categoría
Se añadieron celdas de breakdown de accuracy por categoría tanto para el modelo base como para el fine-tuned, permitiendo identificar en qué categorías el modelo mejora más.

---

## 4. `notebooks/01_eda.ipynb` — Análisis ampliado

### Cambios
- Dataset cambiado a `dataset_train_ds.json` (80 muestras completas)
- Añadidas celdas de análisis de calidad: distribución de `val_context_bool`, `val_stop_reason`
- Añadido análisis de distribución por categoría después del filtrado
- Añadido análisis de longitudes de conversación y detección de multi-turno
- Visualizaciones con matplotlib/seaborn

---

## 5. `notebooks/04_submission.ipynb` — Bug crítico corregido + evaluación local

### Bug corregido: PROMPT_COL
**Problema:** `PROMPT_COL="user_content"` hacía que el pipeline de robustez aplicara el ruido sobre el prompt completo formateado (incluyendo system prompt, historial, rúbrica...) en vez de solo a la última pregunta del usuario.

**Solución:** `PROMPT_COL="question"` — el ruido se aplica solo a `{question}`, que luego se inyecta en la posición correcta del template.

Este era un **bug crítico** que degradaba la métrica de robustez.

### Evaluación local añadida
Se añadió la última celda que:
1. Lee `human_val.verdict_validated` del archivo de validación
2. Calcula Accuracy, Robustez (1 - varianza) y `classification_report` por clase
3. Permite evaluar el modelo localmente antes de la entrega oficial

---

## 6. `docs/metodologia.md` — Documentación para criterios cualitativos

Documento nuevo orientado a los criterios cualitativos del hackathon (60% de la nota):
- Rigor metodológico y eficiencia técnica
- Uso estratégico de datos
- Contribución al análisis del dataset base
- Identificación de sesgos y sugerencias de mejora

---

## 7. `.env` — Variables de entorno configuradas

Archivo creado con todas las variables necesarias para ejecutar el pipeline completo:
- `HF_TOKEN`: token de HuggingFace
- `MODEL_NAME`: `prometheus-eval/prometheus-7b-v2.0`
- `MODEL_FT_PATH`: path al modelo fine-tuned
- `PROMPT_COL=question` (corrección del bug)
- `VALIDATION_INPUT_FILENAME`: path al archivo de validación

El `.env` está en `.gitignore` (no se sube al repositorio por seguridad).

---

## 8. `test_pipeline.py` — Test local sin GPU

Script para verificar que todo el pipeline de datos funciona correctamente sin necesidad de GPU ni modelo cargado:

1. Importaciones de módulos
2. Carga del dataset de entrenamiento
3. Ejecución de `prepare_dataset()`
4. Filtrado con `filter_high_quality()`
5. Generación de prompts con `format_instruction()` y rúbricas por categoría
6. Generación de razonamientos con `generate_reasoning_template()`
7. Bonus: verificación del dataset de validación

```bash
# Ejecutar con:
pip install pandas datasets python-dotenv
python test_pipeline.py
```

---

## Impacto esperado por métrica

| Mejora | Accuracy | Robustez | Contribución |
|--------|----------|----------|--------------|
| Dataset completo (80 vs 10 muestras) | +++ | + | |
| Filtrado de calidad (69 muestras limpias) | ++ | + | ++ |
| LoRA optimizado (r=64, todos módulos, 3 épocas) | ++ | | |
| Rúbricas por categoría en el prompt | ++ | | |
| Historial de conversación separado | + | ++ | |
| Augmentación con ruido (×3 en train) | | +++ | |
| Razonamiento enriquecido en SFT | + | | |
| Bug PROMPT_COL corregido | | ++ | |
| Análisis de sesgos documentado | | | +++ |
| Sugerencias de mejora del dataset | | | ++ |
