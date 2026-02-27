# Documentación del Dataset de Entrada

El dataset proporcionado para los retos del hackathon contiene información detallada sobre interacciones entre usuarios y un modelo de lenguaje, centradas específicamente en escenarios de **seguridad, sesgos y alineación** (Safety & Alignment).

## Formato del Dataset

El dataset se entrega en formato JSON y contiene los siguientes campos principales por registro:

- `iam-id` / `user_id`: Identificadores únicos del usuario que realiza la petición.
- `timestamp`: Fecha y hora de la interacción.
- `message-id`: Identificador único del mensaje.
- `category`: Categoría del riesgo evaluado (ej. Privacidad, Odio, Sesgo de género). Incluye id, icono, nombre y color asociado.
- `challenge`: Descripción de la intención maliciosa o el desafío que el usuario intenta que el modelo incumpla.
- `verdict`: Evaluación previa de la respuesta del modelo (`passed` o `failed`).
  - **passed**: El modelo superó el reto de forma segura.
  - **failed**: El modelo cayó en la trampa (jailbreak) o generó una respuesta insegura.
- `conversation`: Texto completo que contiene todo el historial de interacción. 
- `question`: El último mensaje válido escrito por el usuario en base a `conversation` a efectos de ser evaluado o rebatido.
- `last_interaction`: La respuesta generada por el agente de IA al cual estamos poniendo a prueba como Jueces, sacado de la `conversation`.
- `corrected_response_validated`: Respuesta ideal o segura certificada manualmente (Ground Truth). Este campo actúa como la sugerencia de respuesta correcta.

## Preparación para Evaluación (Prometheus)

Para las tareas del hackathon (como el uso de Prometheus LLM-as-a-Judge o el Fine-Tuning), el pipeline provisto mapea automáticamente los campos hacia el modelo formateador:
1. Convierte el `verdict` original (`passed`/`failed`) a un target binario adecuado para modelaje (`1`/`0`).
2. Mapea la variable `{question}` en el prompt hacia el texto real en la columna `question`.
3. Mapea `{answer}` hacia la columna `last_interaction`. 
4. Mapea `{proposed_answer}` a `corrected_response_validated`.

Recuerda que estas aserciones de variables y el comportamiento de Prometheus puedes inspeccionarlos en vivo desde los notebooks proporcionados como `01_eda.ipynb`.

## Formato de Entrega del Hackathon (Submission)

En el último cuaderno, `04_submission.ipynb`, la herramienta generará de forma automática el `submission.json` a partir del CSV de evaluación (test) ciego que se proporcione. El output final de cada fila tendrá obligatoriamente la siguiente información:

- `"id"`: ID del caso a evaluar.
- `"user_prompt"`: El prompt del usuario usado (extraído de test).
- `"model_pred"`: La clase binaria (`1` passed o `0` failed) que ha decretado tu modelo juez ante la interacción analizada.
- `"model_reason"`: El texto o feedback que generó tu modelo para avalar la decisión anterior.
- `"model_pred_typos"`: La clase binaria frente al test superando variaciones de ruido.
- `"model_reason_typos"`: La razón del modelo frente al test variando el ruido.
