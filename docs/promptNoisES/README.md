# PromptNoisES
**PromptNoisES** is a controllable Spanish linguistic noise generator designed to corrupt **input prompts** for NLP fine-tuning, evaluation and robustness testing.
This tool acts on **short Spanish input text** (such as questions, instructions or conversational turns) to generate three different noisy variants with **separated linguistic effects**.

---

## Key Features
- **Prompt-level**: designed for short inputs such as questions and instructions
- **Three separated noise blocks**:
   1. **Typos** (keyboard mistypings + speed-typing shortcuts)
   2. **Grammatical error** (Spanish-specific rule-based linguistic effects)
   3. **Custom mixed noise** (user-defined)
- **Reproducible generation** via random seeds
- Supports both **JSON** and **CSV** input and output formats

## Design Scope
PromptNoisES is designed to target **Spanish short textual inputs** such as:
- questions
- instructions
- conversational turns

All transformation rules are **explicitly designed for Spanish**, targeting language-specific ortographic and grammatical phenomena.

### Noise blocks

#### Block 1 - Typos
Simulates informal or fast prompt typing, introducing up to 2 errors per input:
- **Keyboard mistypings**
   1. QWERTY-neighbor substitutions
   2. Character omisisions
   3. Abbreviations (`que → q`)
   4. Random space removal (lo hiciste → lohiciste)
- **Speed-typing conventions**
   1. Opening question mark (`¿`) removal
   2. Accent removal (fixed 80% probability)

#### Block 2 - Grammatical errors
Non-probabilistic, simulates rule-based grammatical and ortographic errors. Introduces up to 4 errors per input:
- **Wrong verb form**
   1. *había → habían*
   2. *hemos → habemos*
- **Homophone confusion**
   1. *hecho ↔ echo*
   2. *vaya ↔ valla*
   3. *haber ↔ a ver*
   4. *hay ↔ ay*
   5. *oye ↔ olle*
- **'Porque/Por qué'**
  - *por qué ↔ porque*
- ***Seseo/ceceo***
  - za/ce/ci/zo/zu ↔ sa/se/si/so/su
- **2 person singular preterite form + /s/**
  - From a list of verbs tipically used in assistant-user conversations
  - *dijiste → dijistes*
- **Initial h- drop**
  - *hospital → ospital*
- **B/v swap**
  - If one word contains two b/v characters, swaps the first one
  - *probable → provable*

#### Block 3 - Custom mixed noise
Fully configurable via YAML, this block allows the user to:
- Define the **exact number of typos**
- Define the **exact number of grammatical errors**
- Assign **custom weights to each error type**
- Control **normalization steps** (accent removal, lowercase, punctuation removal)

---

## Input format
You can transform both JSON files and CSV files, with the condition that both must have a `prompt` key or column with the content you want to be transformed.


## Output format
Each input produces 3 output versions:

```json
{
   "prompt_original": "...",
   "prompt_typos": "...",
   "prompt_grammatical_errors": "...",
   "prompt_custom": "..."
}
```

---

## Usage

### JSON

```
python promptnoises.py \
  --input_json input.json \
  --output_json output.json \
  --custom_config custom_config.yaml \
  --seed 42
```

### CSV

```
python promptnoises.py \
  --input_csv input.csv \
  --output_csv output.csv \
  --custom_config custom_config.yaml \
  --seed 42
```

---

## Reproducibility

PromptNoisES supports deterministic generation through random seeds:

```
--seed 42
```

Using the same script version, configuration file, and seed will always produce identical outputs.

---

## Intended Use Cases

- Robustness evaluation
- Sensitivity analysis to orthographic and grammatical noise
- Controlled benchmarking
- Prompt editing for supervised fine-tuning and alignment.

