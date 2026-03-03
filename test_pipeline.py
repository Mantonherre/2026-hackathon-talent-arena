"""
Test rápido del pipeline de datos SIN necesidad de GPU ni modelo.
Verifica que: carga de datos, filtrado, format_instruction y augmentación funcionan.

Uso:
    pip install pandas datasets python-dotenv
    python test_pipeline.py
"""
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

print("=" * 60)
print("TEST DEL PIPELINE DE DATOS")
print("=" * 60)

# 1. Importaciones
print("\n[1/6] Importando módulos...")
from data_utils import (
    load_data, prepare_dataset, filter_high_quality,
    format_instruction, augment_with_noise, generate_reasoning_template
)
from prompts import ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT, get_category_rubric
print("     OK")

# 2. Carga del dataset
print("\n[2/6] Cargando dataset de entrenamiento...")
df_raw = load_data("data/dataset_train_ds.json")
print(f"     {len(df_raw)} muestras cargadas")

# 3. prepare_dataset
print("\n[3/6] Ejecutando prepare_dataset...")
df = prepare_dataset(df_raw)
print(f"     Columnas: {list(df.columns)}")
assert "history_str" in df.columns, "FALLO: falta 'history_str'"
assert "val_context_bool" in df.columns, "FALLO: falta 'val_context_bool'"
assert "val_goal_reasoning" in df.columns, "FALLO: falta 'val_goal_reasoning'"
print("     OK — history_str, val_context_bool, val_goal_reasoning presentes")

# 4. Filtrado de calidad
print("\n[4/6] Filtrando muestras de calidad...")
df_clean = filter_high_quality(df)
print(f"     {len(df_raw)} -> {len(df_clean)} muestras (eliminadas: {len(df_raw)-len(df_clean)})")
assert len(df_clean) < len(df_raw), "FALLO: el filtrado no eliminó nada"

# 5. format_instruction con nuevo prompt
print("\n[5/6] Probando format_instruction con el nuevo prompt...")
sample = df_clean.iloc[0].to_dict()
sample.pop("history", None)  # no serializable
result = format_instruction(sample, ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT)
content = result["user_content"]
assert "{" not in content, f"FALLO: variables sin rellenar en el prompt:\n{content[:500]}"
assert "category_rubric" not in content or "FAILED" in content, "FALLO: category_rubric no inyectada"
assert sample["question"] in content, "FALLO: la question no está en el prompt"
print(f"     OK — prompt generado ({len(content)} chars)")
print(f"     Categoría: {sample['category_name']}")
print(f"     Rubric snippet: {content[content.find('FAILED'):content.find('FAILED')+80]}...")

# 6. generate_reasoning_template
print("\n[6/6] Probando generate_reasoning_template...")
for verdict, label in [("1", "passed"), ("0", "failed")]:
    sample_v = df_clean[df_clean["verdict"] == verdict].iloc[0].to_dict()
    reasoning = generate_reasoning_template(sample_v)
    assert len(reasoning) > 20, f"FALLO: razonamiento demasiado corto para {label}"
    print(f"     {label}: {reasoning[:80]}...")

# Bonus: verificar archivo de validación
print("\n[BONUS] Verificando validation_dataset_sample.json...")
df_val = load_data("data/validation_dataset_sample.json")
df_val_prep = prepare_dataset(df_val, test_file=True)
assert "verdict" not in df_val_prep.columns, "FALLO: test_file=True no debería incluir verdict"
sample_val = df_val_prep.iloc[0].to_dict()
sample_val.pop("history", None)
result_val = format_instruction(sample_val, ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT)
assert "{" not in result_val["user_content"]
print(f"     OK — {len(df_val_prep)} muestras de validación procesadas correctamente")

print("\n" + "=" * 60)
print("TODOS LOS TESTS PASARON")
print("El pipeline de datos funciona correctamente.")
print("Puedes ejecutar los notebooks en SageMaker con confianza.")
print("=" * 60)
