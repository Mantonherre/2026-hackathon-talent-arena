from datasets import Dataset
import pandas as pd
from model_utils import model_predict_batched, split_model_reason_result
from promptnoises import CustomConfig, process_prompts
from data_utils import format_instruction
from prompts import ABS_SYSTEM_PROMPT, ABSOLUTE_PROMPT

def create_robustness_dataset(df_input: pd.DataFrame = None, input_col="user_content") -> pd.DataFrame:
    """
    Crea un dataset con distintas variaciones de ruido o corrupción 
    (typos, errores gramaticales, etc.) aplicadas a los prompts originales
    para evaluar la robustez del modelo.
    
    Args:
        df_input (pd.DataFrame): DataFrame o Dataset de entrada con los prompts originales.
        input_col (str): Nombre de la columna que contiene los prompts. Por defecto "user_content".

    Returns:
        pd.DataFrame: DataFrame de pandas con las variaciones generadas.
    """
    # 1. Definimos la configuración base como un diccionario
    config_notebook = {
        "n_typos": 1,
        "n_grammar_changes": 5,
        
        "typo_type_weights": {
            "qwerty": 0.55,
            "omission": 0.25,
            "abbr": 0.20,
            "space_remove": 0.10
        },
        
        "vowel_delete_bias": 0.9,
        "abbr_q_weight": 0.6,
        "abbr_pq_weight": 0.4,
        
        "grammar_rule_weights": {
            "habia_to_habian": 1.0,
            "hemos_to_habemos": 0.9,
            "homophones": 0.8,
            "porque": 0.9,
            "seseo_ceceo": 0.9,
            "preterite_s": 0.9,
            "drop_initial_h": 0.9,
            "swap_bv": 0.9
        },
        
        "remove_open_questions": True,
        "strip_accents": True,
        "remove_commas": True,
        "lowercase": True
    }

    df_input = df_input.to_pandas() if isinstance(df_input, Dataset) else df_input

    prompts = df_input[input_col].to_list()

    # Usamos process_prompts en lugar de process_csv ya que así evitamos problemas
    # de compatibilidad si el dataset de entrada es un JSON o ya viene cargado en pandas.
    custom_cfg = CustomConfig(**config_notebook)
    results = process_prompts(prompts, custom_cfg=custom_cfg)
    
    df = pd.DataFrame(results)
    return df

def model_preds(model, tokenizer, dataset: Dataset, input_col: str, output_suffix: str) -> Dataset:
        """
        Genera las predicciones para una columna en específico del dataset y formatea la salida
        (razonamiento vs puntuación).

        Args:
            model: Modelo a utilizar (Prometheus, etc.).
            tokenizer: Tokenizador del modelo.
            dataset (Dataset): Dataset con los inputs.
            input_col (str): Nombre de la columna de entrada.
            output_suffix (str): Sufijo para nombrar la columna de salida.

        Returns:
            Dataset: Dataset con las nuevas columnas calculadas.
        """
        

        completion_colname = f"{output_suffix}_completion"

        dataset = dataset.map(
            lambda batch: model_predict_batched(batch = batch, model = model, tokenizer = tokenizer,  input_col = input_col, completion_colname = completion_colname), 
            batched=True, 
            batch_size=8
        )
        
        dataset = dataset.map(
            split_model_reason_result, 
            fn_kwargs={"output_suffix": output_suffix, "input_col": completion_colname}
        )
        return dataset

def format_to_instruction_in_robustness_dataset(robustness_dataset: Dataset, input_col: str = "prompt_original") -> Dataset:
    """
    Formatea todas las instrucciones en el dataset de robustez.

    Args:
        robustness_dataset (Dataset): Dataset con las variaciones.
        input_col (str): Nombre de la columna de entrada. Por defecto "prompt_original".

    Returns:
        Dataset: Dataset con las instrucciones formateadas.
    """
    

    column_mapping = { input_col:"question"}

    robustness_dataset = robustness_dataset.map(format_instruction, fn_kwargs={"system_prompt": ABS_SYSTEM_PROMPT, "user_prompt":ABSOLUTE_PROMPT, "output_col" : input_col, "column_mapping": column_mapping})

    return robustness_dataset

def model_preds_robustness(model, tokenizer, dataset: Dataset, prompt_col: str = "question") -> Dataset:
        """
        Aplica predicción de modelo sobre todas las columnas con variaciones 
        (original, typos, grammatical errors) dentro del dataset.

        Args:
            model: Modelo de inferencia.
            tokenizer: Tokenizador.
            dataset (Dataset): Dataset inicial de donde se generarán las variaciones de robustez.
            prompt_col (str): Nombre de la columna que contiene el prompt. Por defecto "question".

        Returns:
            Dataset: Dataset con los resultados e inferencias generadas para cada variación.
        """

        # convertimos dataset a pandas
        dataset = dataset.to_pandas() if isinstance(dataset, Dataset) else dataset

        # creamos dataset de robustez y mergeamos con el dataset original
        robustness_dataset = create_robustness_dataset(df_input = dataset, input_col = prompt_col)
        
        robustness_dataset = robustness_dataset.merge(dataset, left_on="prompt_original", right_on=prompt_col, how="inner")
        robustness_dataset = Dataset.from_pandas(robustness_dataset)
        
        # formateamos todas las instrucciones
        cols = ["prompt_original", "prompt_typos", "prompt_grammatical_errors"]
        # creamos sufijos para las columnas de salida
        create_suffix = lambda x: "".join([p[0] for p in x.split("_")[:2]]) + "_m"
        col_and_suffix = [(col, create_suffix(col)) for col in cols]

        # aplicamos modelo a cada columna
        for col, output_suffix in col_and_suffix:
            robustness_dataset = format_to_instruction_in_robustness_dataset(robustness_dataset = robustness_dataset, input_col = col)

            robustness_dataset = model_preds(model, tokenizer, robustness_dataset, col, output_suffix)


        
        return robustness_dataset