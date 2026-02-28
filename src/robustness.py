from datasets import Dataset
import pandas as pd
from model_utils import model_predict_batched, split_model_reason_result
from promptnoises import CustomConfig, process_prompts

def create_robustness_dataset(df_input: pd.DataFrame = None, input_col="user_content") -> Dataset:
    """
    Crea un dataset con distintas variaciones de ruido o corrupción 
    (typos, errores gramaticales, etc.) aplicadas a los prompts originales
    para evaluar la robustez del modelo.
    
    Args:
        df_input (pd.DataFrame): DataFrame de entrada con los prompts originales.
        pred_col (str): Nombre de la columna que contiene los prompts. Por defecto "user_content".

    Returns:
        Dataset: Dataset de HuggingFace con las variaciones generadas.
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
    dataset = Dataset.from_pandas(df)
    return dataset

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

def model_preds_robustness(model, tokenizer, dataset: Dataset, input_col: str = "user_content", id_col: str = None) -> Dataset:
        """
        Aplica predicción de modelo sobre todas las columnas con variaciones 
        (original, typos, grammatical errors) dentro del dataset.

        Args:
            model: Modelo de inferencia.
            tokenizer: Tokenizador.
            dataset (Dataset): Dataset generado previamente que contiene las variaciones.
            input_col (str): Nombre de la columna de entrada.

        Returns:
            Dataset: Dataset con los resultados e inferencias generadas.
        """
        robustness_dataset = create_robustness_dataset(df_input = dataset, input_col = input_col)
        
        if id_col:
            id_values = dataset[id_col] if isinstance(dataset, Dataset) else dataset[id_col].tolist()
            robustness_dataset = robustness_dataset.add_column(id_col, id_values)
        else:
            robustness_dataset = robustness_dataset.add_column("id", list(range(len(robustness_dataset))))

        cols = ["prompt_original", "prompt_typos", "prompt_grammatical_errors"]
        
        create_suffix = lambda x: "".join([p[0] for p in x.split("_")[:2]]) + "_m"
        col_and_suffix = [(col, create_suffix(col)) for col in cols]

        for col, output_suffix in col_and_suffix:
            robustness_dataset = model_preds(model, tokenizer, robustness_dataset, col, output_suffix)
        
        return robustness_dataset