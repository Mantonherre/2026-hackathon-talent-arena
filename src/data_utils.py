import pandas as pd
import re
import json
from datasets import load_dataset, Dataset
import os
import string


def load_data(file_path, **args):
    """
    Carga los datos desde un archivo JSON.
    
    Esta función es el primer paso para procesar los datos de entrada del hackathon.
    Te permite importar los datos brutos a un DataFrame de Pandas para su fácil manipulación.
    
    Args:
        file_path (str o Path): Ruta al archivo JSON.
        
    Returns:
        pd.DataFrame: DataFrame que contiene los datos listos para el análisis.
    """
    try:
        return pd.read_json(file_path,**args)
    except Exception as e:
        print(f"Error loading data: {e}")
        return None


def map_verdict(verdict_series):
    """
    Normaliza y mapea la variable verdict a un formato binario estándar ('1' para passed/seguro, '0' para failed/hackeado).
    
    Args:
        verdict_series (pd.Series): Serie de pandas con los veredictos en bruto.
        
    Returns:
        pd.Series: Serie con valores '1', '0' o cadena vacía si no hay coincidencia.
    """
    mapping = {
        "passed": "1", "1": "1", "seguro": "1",
        "failed": "0", "0": "0", "hackeado": "0"
    }
    return (
        verdict_series
        .astype(str)
        .str.lower()
        .str.strip()
        .map(mapping)
        .fillna("")
    )


def prepare_dataset(df, test_file=False):
    """
    Prepara y estructura el dataset crudo para las pruebas del hackathon.
    
    Esta función se encarga de:
    1. Extraer el último turno válido de la conversación entre el usuario y el asistente.
    2. Rellenar las respuestas propuestas ('proposed_answer') si están vacías con la respuesta final.
    3. Mapear el 'verdict' ('passed'/'failed') a un formato binario (1 o 0), omitiéndolo en tests.
    
    Args:
        df (pd.DataFrame): DataFrame con los datos crudos.
        test_file (bool): Si es True, no se procesa ni se incluye la columna 'verdict'.
        
    Returns:
        pd.DataFrame: DataFrame estructurado listo para análisis o entrenamiento.
    """
    # 1. Extraer el último intercambio válido (question, answer, history, conversation)
    # .apply(pd.Series) convierte el diccionario retornado en columnas
    qa_turns = df["raw"].apply(lambda x: get_last_valid_turn(x.get("messages", []))).apply(pd.Series)
    
    # 2. Extraer nombre de la categoría de forma segura
    df["category_name"] = df["category"].apply(lambda x: x.get("name") if isinstance(x, dict) else None)

    # 3. Definir las columnas base que queremos mantener
    # 'challenge', 'category_name' y 'proposed_answer' son fundamentales
    cols_to_keep = ["message-id", "challenge", "category_name", "proposed_answer"]
    
    # Si no es un fichero de test y existe la columna verdict, la incluimos
    if not test_file and "verdict" in df.columns:
        cols_to_keep.insert(0, "verdict")
    
    # 4. Filtrar por las columnas que realmente existan y concatenar con los turnos de QA
    present_cols = [c for c in cols_to_keep if c in df.columns]
    processed_df = pd.concat([df[present_cols], qa_turns], axis=1)
    
    # 5. Rellenar 'proposed_answer' con 'answer' si la primera está vacía
    if "proposed_answer" in processed_df.columns and "answer" in processed_df.columns:
        processed_df["proposed_answer"] = processed_df["proposed_answer"].fillna(processed_df["answer"])
    
    # 6. Mapear veredicto a '1' (passed) o '0' (failed) si corresponde
    if not test_file and "verdict" in processed_df.columns:
        processed_df["verdict"] = map_verdict(processed_df["verdict"])
    
    return processed_df

def save_data(data, file_path):
    """
    Guarda los datos procesados en un archivo JSON.
    
    Utiliza esta función para persistir tus DataFrames o listas de diccionarios después de procesarlos,
    generando los archivos de salida necesarios para las entregas del hackathon.
    
    Args:
        data (lista o pd.DataFrame): Los datos que deseas guardar.
        file_path (str o Path): La ruta de destino del archivo JSON a crear.
    """
    try:
        data = data if isinstance(data, pd.DataFrame) else data.to_pandas()
        data.to_json(file_path, orient='records',
                                 indent=4,
                                 force_ascii=False)
        print(f"Data saved to {file_path}")
    except Exception as e:
        print(f"Error saving data: {e}")

def message_to_conversation_str(history, question=""):
    """
    Convierte una lista de mensajes a un string de conversación previa a la respuesta del modelo.
    
    Args:
        history (list): Lista de diccionarios con claves 'role' y 'content'.
        question (str, opcional): Pregunta final del usuario.

    Returns:
        str: String de conversación previa a la respuesta del modelo.
    """
    content = "\n".join([f"{m.get('role', '').capitalize()}: {m.get('content', '')}" for m in history])
    if question:
        content += f"\nUser: {question}"
    return content
    

def get_last_valid_turn(messages):
    """
    Extrae el último intercambio válido entre el usuario y el asistente de una lista de mensajes.
    
    Busca de atrás hacia adelante el último par donde el rol sea 'assistant' precedido por 'user'.
    Valida que ambos mensajes tengan contenido real y extrae el historial previo para contexto.

    Args:
        messages (list): Lista de diccionarios con claves 'role' y 'content'.

    Returns:
        dict: Diccionario con 'question', 'answer' y 'history', o None si no se encuentra un par válido.
    """
    if not isinstance(messages, list) or len(messages) < 2:
        return None

    for i in range(len(messages) - 1, 0, -1):
        assistant_msg = messages[i]
        user_msg = messages[i-1]
        
        # Validación de roles y contenido no vacío
        if (assistant_msg.get("role") == "assistant" and 
            user_msg.get("role") == "user" and
            assistant_msg.get("content", "").strip() and 
            user_msg.get("content", "").strip()):
            
            return {
                "question": user_msg["content"].strip(),
                "answer": assistant_msg["content"].strip(),
                "history": messages[:i-1],
                "conversation": message_to_conversation_str(messages[:i-1], user_msg["content"].strip())
            }
    return None




def extract_prompt_variables(sample, user_prompt, column_mapping=None):
    """
    Identifica las variables requeridas en una plantilla de prompt y las extrae del sample.
    Permite el uso de un diccionario de mapeo para relacionar nombres del sample con el prompt.

    Args:
        sample (dict o pd.Series): Un ejemplo del dataset que contiene las variables.
        user_prompt (str): La plantilla de prompt que usa llaves {var}.
        column_mapping (dict, opcional): Diccionario que mapea las claves del sample a las 
                                         claves del prompt. Ej: {"nombre_sample": "nombre_prompt"}.

    Returns:
        dict: Diccionario cerrado con únicamente las variables requeridas por el prompt.
    """
    # Identificar las variables que pide la plantilla de forma dinámica
    vars_in_prompt = [fname for _, fname, _, _ in string.Formatter().parse(user_prompt) if fname is not None]
    
    # Invertir el mapeo internamente: {nombre_en_prompt: nombre_en_sample}
    # Esto facilita buscar qué columna del sample corresponde a la variable pedida.
    prompt_to_sample = {}
    if column_mapping:
        prompt_to_sample = {v: k for k, v in column_mapping.items()}
    
    base_vars = {}
    
    # Extraer las variables del sample, validando que existan (ya sea directo o por mapeo)
    for var in vars_in_prompt:
        # Si la variable está mapeada, usamos la clave original del sample; si no, buscamos la variable tal cual
        sample_key = prompt_to_sample.get(var, var)
        
        if sample_key not in sample:
            raise KeyError(
                f"Error: El prompt requiere '{var}' pero la clave '{sample_key}' no está en el sample."
            )
            
        base_vars[var] = sample[sample_key]
        
    return base_vars

def format_instruction(sample, system_prompt, user_prompt, output_col="user_content", column_mapping=None):
    """
    Construye el prompt estructurado para el modelo Prometheus (LLM-as-a-Judge).
    
    Extrae las variables del prompt dinámicamente desde el sample, permitiendo un 
    mapeo de nombres entre ambos.

    Args:
        sample (dict o pd.Series): Un ejemplo del dataset que contiene las variables.
        system_prompt (str): El prompt del sistema general.
        user_prompt (str): La plantilla de prompt que usa llaves {var}.
        output_col (str): Clave de salida.
        column_mapping (dict, opcional): Diccionario de mapeo {"sample_key": "prompt_key"}.

    Returns:
        dict: Diccionario con la clave lista para ser procesada por el tokenizer.
    """
    # Pasamos el mapeo a la función de extracción
    base_vars = extract_prompt_variables(sample, user_prompt, column_mapping)
            
    # Inyección en la plantilla de evaluación absoluta
    user_content = system_prompt + "\n\n" + user_prompt.format(**base_vars)
    
    return {output_col: user_content}


def prepare_sft_binary_text(sample, tokenizer_eos_token='</s>', output_col_name="prompt_sft", 
                            input_col_name="user_content", reasoning_col_name="val_goal_reasoning",
                            label_col_name="verdict"):
    """
    Prepara una muestra de datos para el Supervised Fine-Tuning (SFT) de Prometheus.
    """
    prompt = sample.get(input_col_name, "").strip()
    raw_verdict = sample.get(label_col_name)
    
    # Extraemos el razonamiento. Si por alguna razón está vacío, ponemos un texto genérico de respaldo
    # para no romper el formato de entrenamiento.
    reasoning = sample.get(reasoning_col_name)
    if not reasoning:
        reasoning = "The response is evaluated based on the provided rubric."
    else:
        reasoning = reasoning.strip()
    
    # 1. Normalizar el veredicto
    if isinstance(raw_verdict, str):
        raw_verdict = raw_verdict.strip().lower()

    # 2. Mapeo estricto a binario
    mapping = {
        1: "1", 0: "0", 
        "1": "1", "0": "0",
        "passed": "1", "failed": "0" 
    }
    
    label = mapping.get(raw_verdict)
    
    # 3. Manejo seguro de nulos ANTES de concatenar
    # Si label es None, devolveríamos "[RESULT] None", lo cual contaminaría el entrenamiento.
    if label is None:
        return {output_col_name: ""} 
    
    # 4. El formato para Prometheus
    full_text = f"{prompt}{reasoning} [RESULT] {label}{tokenizer_eos_token}"
    
    return {output_col_name: full_text}