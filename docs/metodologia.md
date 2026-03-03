# Metodología del Equipo — Hackathon ALIA Talent Arena 2026

## Resumen Ejecutivo

Desarrollamos un sistema LLM-as-a-Judge basado en Prometheus-7b-v2.0 con fine-tuning LoRA optimizado para detectar interacciones AI inseguras en español. Implementamos mejoras en tres ejes: **calidad del prompt**, **curación y augmentación de datos**, y **configuración de entrenamiento**.

---

## 1. Rigor Metodológico y Eficiencia Técnica

### 1.1 Prompt Engineering

**Problema identificado:** El prompt original usaba rúbricas genéricas que no distinguían entre tipos de ataque. Un jailbreak de privacidad requiere criterios distintos a uno de odio.

**Solución:** Implementamos rúbricas **por categoría** (`src/prompts.py`):
- `Odio` → foco en contenido degradante o deshumanizante
- `Privacidad` → foco en revelación de datos personales
- `Sesgo de género` → foco en validación de discriminación
- `Delitos no violentos` → foco en asistencia a actividades ilegales
- `Difamación` → foco en afirmaciones falsas sobre personas reales
- `Asesoramiento especializado` → foco en consejo profesional sin supervisión

**Mejora adicional:** El prompt separó explícitamente el historial de conversación (`{history_str}`) de la última pregunta del usuario (`{question}`). Esto permite:
1. Mostrar el contexto multi-turno del ataque
2. Aislar la variante ruidosa (typos/gramática) en `{question}` durante la evaluación de robustez

### 1.2 Evaluación Multi-Escenario

Se ejecuta el modelo sobre tres variantes de cada muestra:
- **Original (po):** texto limpio
- **Typos (pt):** errores de teclado y omisión de letras
- **Gramatical (pg):** errores morfosintácticos (seseo, ceceo, b/v, etc.)

La robustez se mide como `(1 - Varianza) × 100`, donde Varianza = % de casos donde las tres predicciones no coinciden.

### 1.3 Configuración LoRA Optimizada

| Parámetro | Original | Nuestro | Justificación |
|-----------|----------|---------|---------------|
| `r` (rango) | 16 | **64** | Mayor capacidad expresiva para capturar matices semánticos |
| `lora_alpha` | 32 | **128** | Proporcional al rango (ratio 2:1) |
| `target_modules` | q, v | **q, k, v, o, gate, up, down** | Cubre 100% de los parámetros entrenables de Mistral/Llama |
| `num_train_epochs` | 1 | **3** | Suficiente para dataset pequeño sin overfitting |
| `learning_rate` | default | **2e-4** | Óptimo documentado para LoRA en modelos 7B |
| `lr_scheduler` | — | **cosine** | Decaimiento suave para converger sin oscilaciones |
| `weight_decay` | 0 | **0.01** | Regularización L2 contra overfitting |

Carga computacional en A100 80GB: estimado 20-30 min por época con augmentación.

---

## 2. Uso Estratégico de Datos

### 2.1 Fuente y Análisis del Dataset

- **Dataset completo:** 80 muestras de `dataset_train_ds.json` (vs. 10 del sample original)
- **Balance perfecto:** 40 `passed` / 40 `failed`
- **6 categorías:** Odio (19), Delitos no violentos (18), Difamación (17), Asesoramiento especializado (15), Sesgo de género (7), Privacidad (4)

### 2.2 Filtrado de Calidad

Identificamos que **11 muestras** (13.75%) tienen `val_context_bool=False`, indicando que la conversación no es pertinente al reto evaluado. Estas muestras representan "ruido de etiquetado" que penaliza el entrenamiento.

**Criterio de exclusión:** `val_context_bool == False` → `stop_reason = "Context Invalid"`

Resultado: **69 muestras de alta calidad** para entrenamiento (86.25% del total).

### 2.3 Augmentación para Robustez

Para mejorar el score de robustez, augmentamos el dataset de entrenamiento con variantes ruidosas:

```
69 muestras originales × 3 = 207 muestras totales
```

- **Typos (×1):** errores de teclado QWERTY, omisión de letras, abreviaciones
- **Gramática (×1):** seseo/ceceo, b/v, homophones, h inicial eliminada

**Split estratégico:** Solo las muestras originales van al test set (no contaminación). Todas las variantes ruidosas van al train set, maximizando el aprendizaje de robustez.

### 2.4 Razonamiento en SFT

El SFT original usaba `"The response is evaluated based on the provided rubric."` como feedback (placeholder vacío).

Mejoramos esto con `generate_reasoning_template()`:
1. **Primera prioridad:** usar `val_goal_reasoning` del dataset si tiene >30 caracteres (razonamiento validado por humanos)
2. **Fallback:** template estructurado basado en categoría, veredicto y challenge que genera feedback coherente y específico

---

## 3. Contribución al Dataset Base

### 3.1 Análisis de Sesgos Identificados

**Desbalance por categoría:** Las categorías `Privacidad` (4 muestras) y `Sesgo de género` (7 muestras) están significativamente sub-representadas respecto a `Odio` (19) y `Delitos no violentos` (18). Un modelo entrenado sin compensación podría ser menos preciso para casos de privacidad y sesgo de género.

**Cobertura multi-turno:** El 60-70% de las conversaciones son multi-turno (>1 par user/assistant). Esto indica que los atacantes usan estrategias graduales. El modelo base solo evaluaba el último turno — nuestra mejora incluye el historial completo.

**Calidad de `proposed_answer`:** El 23.75% de las muestras tienen `proposed_answer=NaN` (campo vacío). En esos casos rellenamos con la propia respuesta del modelo evaluado (`answer`), lo que da una señal correcta solo cuando la respuesta es segura (passed). Para muestras `failed`, esto introduce una referencia incorrecta. Recomendamos: completar el `proposed_answer` con respuestas seguras para todas las muestras `failed`.

### 3.2 Problemas de Etiquetado

- **Context Invalid (11 muestras):** Las conversaciones no eran pertinentes al challenge declarado. El modelo puede haber recibido etiquetas `passed`/`failed` basadas en premisas incorrectas.
- **Proposed Answer Invalid (1 muestra):** La respuesta de referencia fue marcada como inválida.

### 3.3 Sugerencias de Mejora del Dataset

| Problema | Impacto | Sugerencia |
|----------|---------|-----------|
| Sub-representación de Privacidad/Sesgo género | Menor precisión en esas categorías | Añadir 15-20 muestras por categoría |
| `proposed_answer` vacío en muestras failed | Señal de referencia incorrecta en SFT | Completar con respuestas seguras curadas por humanos |
| Context Invalid no filtrado | Ruido en entrenamiento | Excluir automáticamente o re-etiquetar |
| Falta de ataques multi-modelo | Benchmark limitado | Incluir respuestas de diferentes LLMs para el mismo challenge |

---

## 4. Resumen de Mejoras

| Mejora | Criterio Impactado | Impacto Esperado |
|--------|-------------------|-----------------|
| Dataset completo (80 vs 10) | Accuracy | +++ |
| Filtrado de calidad (69 muestras) | Accuracy + Datos | ++ |
| LoRA r=64, todos los módulos, 3 épocas | Accuracy | ++ |
| Rúbricas por categoría | Accuracy | ++ |
| Historial de conversación en prompt | Accuracy | + |
| Augmentación con ruido (×3) | Robustez | +++ |
| Razonamiento enriquecido en SFT | Accuracy | + |
| Análisis de dataset (contribución) | Contribución | ++ |
