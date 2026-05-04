# CF-architecture

Repository containing code for CF architecture and experiments presented on the paper "Enhancing Question-Answering Factuality: Conformal Prediction in the Cocoruta 2.0 Model".

## Set-up

First, create a Python virtual environment and install the requirements:

```bash
pip install -r requirements.txt
```

Then, add your Open-AI key to the environment variables:

```bash
export OAI_KEY=[youy-openai-key]
```

## Running the CF-architecture (calibration + control)

1) Activate your virtual environment

2) Add a file with questions in `data/<prefix>_evaluation.txt`. It must contain questions for calibration and control steps (the default is 50 questions for calibration sampled randomly from the file, the remaining questions are used in control step).

3) Run the subclaims decomposition (DECOMPOSE_SUBCLAIMS action):

```python
python src/main.py --action DECOMPOSE_SUBCLAIMS --specialist_model felipeoes/cocoruta-2-8b --generalist_model gpt-5-mini --input_filename data/cocoruta_evaluation.txt --embedding_model text-embedding-3-large
```

4) Copy the created file `out/<prefix>_subclaims.jsonl` to `data/<prefix>_annotation.jsonl`

5) Run the threshold calibration (CALIBRATE action):

```python
python src/main.py --action CALIBRATE --specialist_model felipeoes/cocoruta-2-8b --generalist_model gpt-5-mini --embedding_model text-embedding-3-large --annotation_filename data/cocoruta_annotation.jsonl --alpha 0.2 --a 1.0 --confidence_method frequency-score --alphas "[0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]" --confidence_methods "['baseline-score', 'random-score', 'model-score', 'average-model-score', 'frequency-score', 'factual']"
```

6) Run the control over new examples (CONTROL action):

```python
python src/main.py --action CONTROL --specialist_model felipeoes/cocoruta-2-8b --generalist_model gpt-5-mini --embedding_model text-embedding-3-large --threshold_filename out/cocoruta_a=1.0_alpha=0.2_conf=frequency-score.pkl --control_filename data/cocoruta_control.jsonl
```

## Running the CF-architecture (only control)

1) Activate your virtual environment

2) Add a file with yout test questions in `data/<prefix>_evaluation.txt`

3) Run the subclaims decomposition (DECOMPOSE_SUBCLAIMS action):

```python
python src/main.py --action DECOMPOSE_SUBCLAIMS --specialist_model felipeoes/cocoruta-2-8b --generalist_model gpt-5-mini --input_filename data/cocoruta_evaluation.txt --embedding_model text-embedding-3-large
```

4) Copy the created file `out/<prefix>_subclaims.jsonl` to `data/<prefix>_control.jsonl`

5) Be sure to have a pickle file with the computed threshold. It has this name by default: `out/<dataset_prefix>_a=<REQUIRED_FRAC_CORRECT>_alpha=<ALPHA>_conf=<CONFIDENCE_METHOD>.pkl`

6) Run the control over new examples (CONTROL action):

```python
python src/main.py --action CONTROL --specialist_model felipeoes/cocoruta-2-8b --generalist_model gpt-5-mini --embedding_model text-embedding-3-large --threshold_filename out/cocoruta_a=1.0_alpha=0.2_conf=frequency-score.pkl --control_filename data/cocoruta_control.jsonl
```
