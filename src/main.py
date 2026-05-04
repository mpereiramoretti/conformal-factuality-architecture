import argparse
import ast
from datetime import datetime
from typing import List
from cf.factual_conformalizer_plotter import FactualConformalizerPlotter
import joblib
import torch
import logging
import os
import random

import numpy as np

from metrics.accepted_subclaims_fraction_metric import AcceptedSubclaimsFractionMetric
from metrics.correlation_cf_metric import CorrelationCFMetric
from metrics.merged_cosine_similarity_metric import MergedCosineSimilarityMetric
from metrics.metric_interface import MetricInterface
from scores.average_model_score import AverageModelScore
from scores.baseline_score import BaselineScore
from scores.frequency_score import FrequencyScore
from scores.model_score import ModelScore
from scores.random_score import RandomScore
random.seed(42)

from tqdm import tqdm

from cf.factuality_annotator import FactualityAnnotator
from cf.factual_conformalizer import FactualConformalizer
from cf.subclaims_decomposer import SubclaimsDecomposer
from cf.subclaims_merger import SubclaimsMerger
from llms.model_factory import ModelFactory
from utils.file_manager import FileManager


INPUT_SIZE = -1  # -1 represents all the lines!
BREAKDOWN_PROMPT = "Por favor, divida a seguinte entrada em um conjunto de pequenas afirmações independentes e retorne a saída como um JSONL, onde cada linha é {subclaim:[AFIRMAÇÃO]}. O tipo de [AFIRMAÇÃO] deve ser string, e não lista. Caso você identifique afirmações repetidas, mantenha somente uma. Caso você identifique afirmações incompletas, remova-as. Remova qualquer menção a números e datas de Lei, Decreto, Decreto-Lei ou Resolução.\n A entrada é:"
FACTUALITY_PROMPT = "Você é um avaliador de factualidade de afirmações. Vou te passar uma questão e uma lista de afirmações sobre a questão apresentada. Retorne SOMENTE um JSONL no formato {subclaim:[SUBCLAIM], score:[SCORE], basis:[BASIS]}. [SUBCLAIM] é uma string contendo o texto da afirmação, não incluir o número de sequência. [ANNOTATION] é um score de factualidade entre 0 e 1, em que 0 representa uma afirmação claramente falsa tal como 'A Terra é plana', e 1 se for claramente factual tal como '1+1=2'. Tente se embasar em leis, decretos-leis, decretos e normas sobre o direito do mar no Brasil sempre que possível. [BASIS] é uma string contendo uma fundamentação jurídica da marcação, escreva uma fundamentação com uma frase simples, contendo o número da lei, se tiver.\n\nA questão é:\n<<PROMPT>>\n\nAs afirmações são:\n<<SUBCLAIMS>>"
MERGE_PROMPT = "Você receberá uma instrução e um conjunto de fatos que são verdadeiros. Construa uma resposta usando APENAS os fatos fornecidos e tente usar todos os fatos sempre que possível. Se nenhum fato for fornecido, responda à instrução incorporando o fato de que você não sabe o suficiente para responder completamente.\n\nOs fatos:\n{claim_string}\n\nA instrução:\n{prompt}"
SCORING_PROMPT = "Eu vou lhe fornecer uma afirmação. Por favor, retorne um score de confiança para essa afirmação.\n Esse score deve representar o seu nível de confiança na afirmação, onde 1,00 corresponde a fatos óbvios como 'O Pará é um estado brasileiro' e 'O Brasil é uma República'. Já 0,00 corresponde a afirmações muito obscuras ou difíceis de serem conhecidas por qualquer pessoa. Retorne SOMENTE o score com duas casas decimais.\n\n A afirmação de saída é: {claim}."
COUNTING_PROMPT = "Você receberá uma lista de afirmações e um trecho de texto. Para cada afirmação, avalie se o texto a apoia, a contradiz ou não está relacionado a ela. Retorne diretamente um JSONL, onde cada linha é {\"id\":[CLAIM_ID], \"score\":[SCORE]}. Retorne diretamente o JSONL sem explicações ou qualquer outra formatação. Para o [SCORE], retorne 1 para apoia, -1 para contradiz e 0 para não relacionado. As afirmações são:\n"
CORRECT_ANNOTATIONS = ["Y", "S"]
AVERAGE_MODEL_SCORE_ITERATIONS = 5
FREQUENCY_SCORE_N_SAMPLES = 5
DEVICE_MAP = "cuda:0"
CALIBRATION_SIZE = 50
N_FOLDS = 10


def setup_logger():
    os.makedirs("logs", exist_ok=True)

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/cf_architecture_{timestamp}.log"

    logging.basicConfig(
        filename=log_filename,
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        encoding="utf-8"
    )


def log_env_config():
    logger = logging.getLogger(__name__)

    try:
        logger.info(f"cuda available: {str(torch.cuda.is_available())}")
        logger.info(f"cuda devices: {str(torch.cuda.device_count())}")
        for i in range(torch.cuda.device_count()):
            logger.info(str(torch.cuda.get_device_name(i)))
        logger.info(f"torch cuda version: {str(torch.version.cuda)}")
        logger.info(str(torch.__version__))
    except Exception as e:
        logger.error(f"Could not collect the CUDA env in this machine. Maybe CUDA is not available? Exception: {e}")


def parse_args():
    parser = argparse.ArgumentParser(description="Conformal Factuality Architecture")
    parser.add_argument("--specialist_model", type=str, required=True)  # e.g. felipeoes/cocoruta-2-8b
    parser.add_argument("--generalist_model", type=str, required=True)  # e.g. gpt-5-mini
    parser.add_argument("--action", type=str, required=True)            # e.g. DECOMPOSE_SUBCLAIMS
    parser.add_argument("--input_filename", type=str)        # mandatory to DECOMPOSE_SUBCLAIMS (e.g. data/cocoruta_evaluation.txt)
    parser.add_argument("--threshold_filename", type=str)    # mandatory to CONTROL (e.g. out/cocoruta_a=1_alpha=0.1_conf=frequency+model.pkl)
    parser.add_argument("--control_filename", type=str)      # mandatory to CONTROL (e.g. data/cocoruta_annotation.jsonl)
    parser.add_argument("--embedding_model", type=str)       # mandatory to MergedCosineSimilarityMetric on CALIBRATION and CONTROL e. g. text-embedding-3-large
    parser.add_argument("--annotation_filename", type=str)   # mandatory to CALIBRATE (e.g. data/cocoruta_annotation.jsonl)
    parser.add_argument("--alpha", type=float)               # mandatory to CALIBRATE
    parser.add_argument("--a", type=float)                   # mandatory to CALIBRATE
    parser.add_argument("--confidence_method", type=str)     # mandatory to CALIBRATE
    parser.add_argument("--alphas", type=str)                # optional to CALIBRATE
    parser.add_argument("--confidence_methods", type=str)    # optional to CALIBRATE
    return parser.parse_args()


def prepare_metrics(dataset: List[dict], cf: FactualConformalizer, sm: SubclaimsMerger):
    logger = logging.getLogger(__name__)
    failed_entries = []
    for i, entry in tqdm(enumerate(dataset)):
        try:
            logger.info(f"#######   ENTRY {entry['index']}   #######")

            logger.info("------- Assigning for each subclaim if it is accepted or not -------")
            cf.mark_as_accepted(entry)
            cf.log_data_stats(entry)  # logging statistics of test data after marking the accepted subclaims
            logger.info("")

            logger.info("------- Merging the accepted subclaims -------")
            entry["merged_accepted"] = sm.merge_accepted(entry)
            logger.info("")

            logger.info("------- Merging the factual subclaims -------")
            entry["merged_factual"] = sm.merge_factual(entry)
            logger.info("")

            logger.info("------- Assigning the quantities of accepted and factual subclaims -------")
            entry["total_subclaims"] = len(entry["claims"])
            entry["total_accepted"] = sum(subclaim["accepted"] == 1 for subclaim in entry["claims"])
            entry["total_factual"] = sum(subclaim["factual"] == 1 for subclaim in entry["claims"])
            entry["total_accepted_and_factual"] = sum((subclaim["accepted"] == 1) and (subclaim["factual"] == 1) for subclaim in entry["claims"])

        except Exception as ex:
            logger.warning(f"Error controlling for entry {entry['index']}: {ex}")
            failed_entries.append(entry['index'])
            logger.warning(f"Continuing with next entry...")

    logger.info(f"Error controlling for these entries: {str(failed_entries)}")
    logger.info(f"")


def compute_metrics(data: List[dict], metrics: List[MetricInterface]) -> dict:
    logger = logging.getLogger(__name__)

    logger.info("------- Computing metrics -------")
    aggregated_metrics = {}
    for metric in metrics:
        metric_name = metric.get_name()
        logger.info(f">>>>> Metric: {metric_name}")
        if metric_name != "correlation-cf-metric":  # computed over all the metrics
            # Compute metric for each entry
            for entry in tqdm(data):
                logger.info(f"Entry: {entry['index']}")
                entry[metric_name] = metric.compute(entry)
                logger.info(f"{entry[metric_name]:.6f}")
        aggregated_metrics[metric_name + "_mean"] = metric.aggregate(data)[0]
        aggregated_metrics[metric_name + "_std"] = metric.aggregate(data)[1]

    logger.info("------- Agregated metrics -------")
    for metric_name, value in aggregated_metrics.items():
        logger.info(f"{metric_name}: {value}")
    logger.info("")

    return aggregated_metrics


def main():
    logger = logging.getLogger(__name__)

    # Parameters of the call
    args = parse_args()
    SPECIALIST_MODEL = args.specialist_model
    GENERALIST_MODEL = args.generalist_model
    ACTION = args.action
    INPUT_FILENAME = args.input_filename
    THRESHOLD_FILENAME = args.threshold_filename
    CONTROL_FILENAME = args.control_filename
    EMBEDDING_MODEL = args.embedding_model
    ANNOTATION_FILENAME = args.annotation_filename
    ALPHA = args.alpha
    REQUIRED_FRAC_CORRECT = args.a
    CONFIDENCE_METHOD = args.confidence_method
    ALPHAS = args.alphas
    CONFIDENCE_METHODS = args.confidence_methods
    logger.info("Parameters of the call:")
    logger.info(f"SPECIALIST_MODEL: {SPECIALIST_MODEL}")
    logger.info(f"GENERALIST_MODEL: {GENERALIST_MODEL}")
    logger.info(f"ACTION: {ACTION}")
    logger.info(f"INPUT_FILENAME: {INPUT_FILENAME}")
    logger.info(f"THRESHOLD_FILENAME: {THRESHOLD_FILENAME}")
    logger.info(f"CONTROL_FILENAME: {CONTROL_FILENAME}")
    logger.info(f"EMBEDDING_MODEL: {EMBEDDING_MODEL}")
    logger.info(f"ANNOTATION_FILENAME: {ANNOTATION_FILENAME}")
    logger.info(f"ALPHA: {ALPHA}")
    logger.info(f"REQUIRED_FRAC_CORRECT: {REQUIRED_FRAC_CORRECT}")
    logger.info(f"CONFIDENCE_METHOD: {CONFIDENCE_METHOD}")
    logger.info(f"ALPHAS: {ALPHAS}")
    logger.info(f"CONFIDENCE_METHODS: {CONFIDENCE_METHODS}")
    logger.info("")

    # Action is None
    if ACTION is None:
        raise ValueError(
            "An action must be set - please set ACTION as DECOMPOSE_SUBCLAIMS, CALIBRATE, or CONTROL (with command: python src/main.py --action [ACTION])"
        )

    # Action has invalid value
    if ACTION not in ["DECOMPOSE_SUBCLAIMS", "CALIBRATE", "CONTROL"]:
        raise ValueError(
            f"The value {str(ACTION)} is invalid for ACTION. ACTION values must be set DECOMPOSE_SUBCLAIMS, CALIBRATE, or CONTROL (with command: python src/main.py --action [ACTION])"
        )

    # Open-AI Key
    logger.info("Loading OpenAI Key...")
    OAI_KEY = os.environ.get("OAI_KEY")
    if OAI_KEY is None:
        raise ValueError(
            "OpenAI key is not set - please set OAI_KEY to your OpenAI key (with command: export OAI_KEY=[OAI_KEY])"
        )
    logger.info("")

    # Load models
    logger.info("Loading models...")
    logger.info(f"--> Using GPU: {DEVICE_MAP}")
    mf = ModelFactory(oai_key=OAI_KEY, device_map=DEVICE_MAP)
    specialist_model = mf.get_model(model_name=SPECIALIST_MODEL)
    generalist_model = mf.get_model(model_name=GENERALIST_MODEL)
    specialist_model.load_model()
    generalist_model.load_model()
    logger.info("")

    # Instantiate the objects
    file_manager = FileManager()
    sd = SubclaimsDecomposer(generalist_model, BREAKDOWN_PROMPT)
    fa = FactualityAnnotator(generalist_model, FACTUALITY_PROMPT)
    sm = SubclaimsMerger(generalist_model, MERGE_PROMPT)

    # Instatiate scores
    # TODO implement ranking scores
    SCORES = [
        BaselineScore(),
        RandomScore(),
        ModelScore(
            specialist_model,
            SCORING_PROMPT
        ),
        AverageModelScore(
            specialist_model,
            SCORING_PROMPT,
            AVERAGE_MODEL_SCORE_ITERATIONS
        ),
        FrequencyScore(
            specialist_model,
            generalist_model,
            COUNTING_PROMPT,
            n_samples=FREQUENCY_SCORE_N_SAMPLES
        ),
    ]

    def load_metrics():
        # Load embedding model
        logger.info("Loading embedding model...")
        if EMBEDDING_MODEL:
            embedding_model = mf.get_model(model_name=EMBEDDING_MODEL)
            embedding_model.load_model()
        else:
            logger.info("The embedding model was not specified...")

        METRICS = [
            AcceptedSubclaimsFractionMetric(),
            CorrelationCFMetric(),
            MergedCosineSimilarityMetric(embedding_model)
        ]
        return METRICS

    ############################################################################################
    ############################################################################################
    ############################################################################################

    if ACTION == "DECOMPOSE_SUBCLAIMS":  # GENERATE OUTPUTS + DECOMPOSE IN SUBCLAIMS
        logger.info("#" * 80)
        logger.info("DECOMPOSING AND LLM-SCORING SUBCLAIMS...")
        logger.info("#" * 80)

        logger.info("")

        try:
            dataset_prefix = INPUT_FILENAME.split("/")[-1].split("_")[0]
            logger.info(f"Dataset prefix: {dataset_prefix}\n")
        except Exception as e:
            raise ValueError(f"Option --input_filename must be passed when action is DECOMPOSE_SUBCLAIMS (with command: python src/main.py --action DECOMPOSE_SUBCLAIMS --input_filename [INPUT_FILENAME]. Exception: {e}")

        logger.info("======== Reading the input file ========")
        logger.info(f"Filename: {INPUT_FILENAME}")
        input_dataset = file_manager.load_input_lines(INPUT_FILENAME)
        if INPUT_SIZE > 0:
            input_dataset = input_dataset[:INPUT_SIZE]
        logger.info(f"Successfully read the file {INPUT_FILENAME} with {len(input_dataset)} lines!")
        file_manager.head(input_dataset, n=5)  # shows the 5 first examples
        logger.info("")

        #logger.info("======== Filtering only failed entries ========")
        #failed = [23, 54, 66]
        #input_dataset = [input_dataset[i] for i in failed]
        #logger.info(f"Successfully filtered only the {len(input_dataset)} failed entries: {str(failed)}")
        #file_manager.head(input_dataset, n=5)  # shows the 5 first examples
        #logger.info("")

        logger.info("======== Iterating on entries ========")
        data = []
        failed_entries = []
        for i in range(len(input_dataset)):
            try:
                logger.info(f"#######   ENTRY {i}   #######")
                #logger.info(f"#######   ENTRY {failed[i]}   #######")

                logger.info("------- Creating JSON -------")
                prompt = input_dataset[i]
                entry = {"index": i, "prompt": prompt}
                #entry = {"index": failed[i], "prompt": prompt}

                logger.info("------- Generating output -------")
                logger.info(f">>> Prompt: {prompt}")
                messages = specialist_model.build_message(prompt)
                output = specialist_model.generate(messages=messages)
                entry["output"] = output
                logger.info(f">>> Output: {output}")

                logger.info("------- Decomposing the outputs into subclaims -------")
                subclaims = sd.decompose_into_subclaims(entry)
                entry["claims"] = [{"subclaim": claim} for claim in subclaims]
                m = len(entry["claims"])
                logger.info(f">>> Entry {i}: Generated {m} subclaims.")

                logger.info("------- Scoring the subclaims -------")
                logger.info("1) Computing random normal noise (to break ties later)...")
                for subclaim in entry["claims"]:
                    subclaim["noise"] = np.random.normal(0, 0.001)

                logger.info("2) Computing scores...")
                for score in SCORES:
                    score_name = score.get_name()
                    logger.info(f">>> Score: {score_name}")
                    if score_name != "frequency-score":
                        for j, subclaim in enumerate(entry["claims"]):
                            subclaim[score_name] = score.compute(entry=entry, subclaim=subclaim)
                            logger.info(f"--> Subclaim {j} score: {str(subclaim[score_name])}")
                    else:
                        logger.info(f"--> n_samples: {FREQUENCY_SCORE_N_SAMPLES}")
                        frequency_scores = score.compute(entry=entry)
                        for j, subclaim in enumerate(entry["claims"]):
                            subclaim[score_name] = frequency_scores[j]
                            logger.info(f"--> Subclaim {j} score: {str(subclaim[score_name])}")
                logger.info(f"Generated scores for entry {i}.")

                #logger.info("------- Assigning 'N' to each subclaim for annotation -------")
                #for subclaim in entry["claims"]:
                #    subclaim["annotation"] = "N"
                logger.info("------- Assigning factuality annotation via LLM-as-a-Judge -------")
                annot_list = fa.get_annotations(entry=entry)
                for j, subclaim in enumerate(entry["claims"]):
                    subclaim["factuality_score"] = annot_list[j]["score"]
                    subclaim["annotation"] = "Y" if annot_list[j]["score"] > 0.5 else "N"
                    subclaim["basis"] = annot_list[j]["basis"]
                    logger.info(f"--> Subclaim {j} factuality: {str(subclaim["factuality_score"])} ({str(subclaim["annotation"])})")
                total_factual = sum(int(subclaim["annotation"] == "Y") for subclaim in entry["claims"])
                total_subclaims = len(entry["claims"])
                pct = np.round(total_factual / total_subclaims * 100, 2) if total_subclaims > 0 else 0
                logger.info(f"Generated annotations for entry {i}. Qtt of factual subclaims: {total_factual} / {total_subclaims} ({pct}%).")

                logger.info("------- Generating file with decomposed subclaims -------")
                data.append(entry)
                jsonl_output = {"data": data}
                file_manager.save_output_to_file(jsonl_output, f"out/{dataset_prefix}_subclaims.jsonl")
                logger.info(f"Successfully saved file out/{dataset_prefix}_subclaims.jsonl")
                #file_manager.save_output_to_file(jsonl_output, f"out/{dataset_prefix}_subclaims_failed.jsonl")
                #logger.info(f"Successfully saved file out/{dataset_prefix}_subclaims_failed.jsonl.")

            except Exception as ex:
                logger.warning(f"Error processing entry {i}: {ex}")
                failed_entries.append(i)
                logger.warning(f"Continuing with next entry...")

            logger.info("")

        logger.info(f"Entries that failed to process ({str(len(failed_entries))}): {str(failed_entries)}")
        logger.info("")
        logger.info(f"Please, copy out/{dataset_prefix}_subclaims_more_basis.jsonl to data/{dataset_prefix}_annotation_more_basis.jsonl and annotate it.\n")

    ############################################################################################
    ############################################################################################
    ############################################################################################

    elif ACTION == "CALIBRATE":  # CALIBRATION + STUDY
        logger.info("#" * 80)
        logger.info("CALIBRATING THE THRESHOLD...")
        logger.info("#" * 80)

        logger.info("")

        try:
            ALPHA = float(ALPHA)
        except:
            raise ValueError(f"Option --alpha must be passed when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --alpha [ALPHA].")

        try:
            REQUIRED_FRAC_CORRECT = float(REQUIRED_FRAC_CORRECT)
        except:
            raise ValueError(f"Option --a must be passed when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --a [REQUIRED_FRAC_CORRECT].")

        if not CONFIDENCE_METHOD:
            raise ValueError(f"Option --confidence_method must be passed when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --confidence_method [CONFIDENCE_METHOD].")

        if not EMBEDDING_MODEL:
            raise ValueError(f"Option --embedding_model must be passed when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --embedding_model [EMBEDDING_MODEL].")

        try:
            ALPHAS = ast.literal_eval(ALPHAS)
        except:
            raise ValueError(f"Option --alphas should be passed as a list of floats when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --alphas [ALPHAS].")

        try:
            CONFIDENCE_METHODS = ast.literal_eval(CONFIDENCE_METHODS)
        except:
            raise ValueError(f"Option --confidence_methods should be passed as a list of strings when action is CALIBRATE (with command: python src/main.py --action CONFIDENCE_METHODS --confidence_methods [CONFIDENCE_METHODS].")

        logger.info("======== Loading metrics ========")
        METRICS = load_metrics()
        logger.info("")

        logger.info("======== Reading the annotated file ========")
        try:
            logger.info(f"Filename: {ANNOTATION_FILENAME}")
            dataset_prefix = ANNOTATION_FILENAME.split("/")[-1].split("_")[0]
        except Exception as e:
            raise ValueError(f"Option --annotation_filename must be passed when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --annotation_filename [ANNOTATION_FILENAME]. Exception: {e}")
        data = file_manager.load_output_from_file(ANNOTATION_FILENAME)
        n = len(data)
        logger.info(f"Successfully read the file {ANNOTATION_FILENAME} with {n} entries!")
        logger.info("")

        logger.info("======== Selecting randomly the calibration set  ========")
        indices = sorted(range(len(data)))
        CALIBRATION_INDICES = sorted(random.sample(indices, CALIBRATION_SIZE))
        CONTROL_INDICES = sorted(set(indices) - set(CALIBRATION_INDICES))
        calibration_data = [entry for entry in data if entry["index"] in CALIBRATION_INDICES]
        m = len(calibration_data)
        logger.info(f"Filtered {m} entries randomly for calibration.")
        file_manager.head(calibration_data, n=5)  # shows the 5 first examples
        logger.info("")

        logger.info("======== Generating the control dataset  ========")
        control_data = [entry for entry in data if entry["index"] in CONTROL_INDICES]
        n = len(control_data)
        logger.info(f"Saving {n} entries randomly to the control step.")
        jsonl_output = {"data": control_data}
        file_manager.save_output_to_file(jsonl_output, f"data/{dataset_prefix}_control.jsonl")
        logger.info(f"Successfully saved file data/{dataset_prefix}_control.jsonl")
        logger.info("")

        logger.info("======== Assigning for each subclaim factuality w.r.t. entailment ========")
        for entry in tqdm(calibration_data):
            for subclaim in entry["claims"]:
                subclaim["factual"] = int(subclaim["annotation"] in CORRECT_ANNOTATIONS)
        logger.info("")

        logger.info("======== Computing the single threshold (full calibration data) ========")
        cf = FactualConformalizer(
            alpha=ALPHA,
            a=REQUIRED_FRAC_CORRECT,
            confidence_method=CONFIDENCE_METHOD
        )
        cf.calibrate_threshold(calibration_data)
        thresh_filename = f"out/{dataset_prefix}_a={str(REQUIRED_FRAC_CORRECT)}_alpha={str(ALPHA)}_conf={CONFIDENCE_METHOD}.pkl"
        joblib.dump(cf, thresh_filename)
        logger.info(f"Threshold saved: at {thresh_filename}")
        logger.info("")

        logger.info("STUDYING CF IN THIS DATASET...")
        logger.info("======== Histogram of removed subclaims (Leave-One-Out) ========")  # single alpha
        cf_plt = FactualConformalizerPlotter(dataset_prefix=dataset_prefix)
        hist_filename = f"out/{dataset_prefix}_hist_a={REQUIRED_FRAC_CORRECT}_alpha={ALPHA}_conf={CONFIDENCE_METHOD}.png"
        cf_plt.plot_hist_loo(
            a=REQUIRED_FRAC_CORRECT,
            alpha=ALPHA,
            confidence_method=CONFIDENCE_METHOD,
            data=calibration_data,
            hist_filename=hist_filename
        )
        logger.info("")

        logger.info("======== Generating merged outputs (Leave-One-Out) ========")  # single alpha
        sm.generate_merged_outputs_loo(
            data=calibration_data,
            alpha=ALPHA,
            a=REQUIRED_FRAC_CORRECT,
            confidence_method=CONFIDENCE_METHOD
        )
        logger.info("")

        if ALPHAS and CONFIDENCE_METHODS:
            logger.info("======== Creating calibration plots for ALPHAS and CONFIDENCE_METHODS (K-Fold) ========")  # guarantee evaluation
            fig_filename = f"out/{dataset_prefix}_raw_calibration_a={REQUIRED_FRAC_CORRECT}.png"
            cf_plt.create_calibration_plot(
                data=calibration_data,
                alphas=ALPHAS,
                a=REQUIRED_FRAC_CORRECT,
                confidence_method=CONFIDENCE_METHOD,
                fig_filename=fig_filename
            )
            logger.info("")

            logger.info("======== Creating correctness plots for ALPHAS and CONFIDENCE_METHODS (Leave-One-Out) ========")  # example by example, then plot
            fig_filename = f"out/{dataset_prefix}_raw_a={REQUIRED_FRAC_CORRECT}_fig.png"
            cf_plt.create_correctness_vs_removed_plot_loo(
                data=calibration_data,
                alphas=ALPHAS,
                a=REQUIRED_FRAC_CORRECT,
                confidence_methods=CONFIDENCE_METHODS,
                fig_filename=fig_filename
            )
        else:
            logger.warning("Both --alphas and --confidence_methods must be passed to generate calibration and correctness plots when action is CALIBRATE (with command: python src/main.py --action CALIBRATE --alphas [ALPHAS] --confidence_methods [CONFIDENCE_METHODS]).")
        logger.info("")

        logger.info("======== Compute metrics over test data (K-Fold) ========")
        logger.info(f"K-fold computation with K = {N_FOLDS}")
        results = {}
        for i in range(N_FOLDS):
            logger.info(f">>> Iteration {i + 1} / {N_FOLDS}")

            # Randomly shuffle the data
            random.shuffle(calibration_data)

            # Split the data into two equal parts
            split_index = len(calibration_data) // 2
            calibr_data = calibration_data[:split_index]
            test_data = calibration_data[split_index:]

            # compute the threshold
            cf = FactualConformalizer(
                alpha=ALPHA,
                a=REQUIRED_FRAC_CORRECT,
                confidence_method=CONFIDENCE_METHOD
            )
            cf.calibrate_threshold(calibr_data)

            # prepare and compute aggregated metrics of this fold
            prepare_metrics(dataset=test_data, cf=cf, sm=sm)
            aggregated_metrics = compute_metrics(data=test_data, metrics=METRICS)
            results[f"Fold {i + 1}"] = aggregated_metrics

            logger.info("------- Generating file with results across folds -------")
            file_manager.save_output_to_file(results, f"out/{dataset_prefix}_calibration_metrics.jsonl")
            logger.info(f"Successfully saved file out/{dataset_prefix}_calibration_metrics.jsonl.")
            logger.info("")

        # plot the metrics distribution over test data
        logger.info("Plotting metrics distribution across folds")
        fig_filename = f"out/{dataset_prefix}_metrics_a={REQUIRED_FRAC_CORRECT}_alpha={ALPHA}_conf={CONFIDENCE_METHOD}.png"
        cf_plt.plot_metrics_distribution(
            results=results,
            k=N_FOLDS,
            fig_filename=fig_filename
        )
        logger.info("")

    ############################################################################################
    ############################################################################################
    ############################################################################################

    elif ACTION == "CONTROL":  # TEST TIME
        logger.info("#" * 80)
        logger.info("CONTROLLING THE FACTUALITY OF TEST DATA...")
        logger.info("#" * 80)

        logger.info("")

        if not THRESHOLD_FILENAME:
            raise ValueError(f"Option --threshold_filename must be passed when action is CONTROL (with command: python src/main.py --action CONTROL --threshold_filename [THRESHOLD_FILENAME].")

        if not CONTROL_FILENAME:
            raise ValueError(f"Option --control_filename must be passed when action is CONTROL (with command: python src/main.py --action CONTROL --control_filename [CONTROL_FILENAME].")

        if not EMBEDDING_MODEL:
            raise ValueError(f"Option --embedding_model must be passed when action is CONTROL (with command: python src/main.py --action CONTROL --embedding_model [EMBEDDING_MODEL].")

        logger.info("======== Loading metrics ========")
        METRICS = load_metrics()
        logger.info("")

        logger.info("======== Reading the threshold file ========")
        try:
            THRESHOLD_FILENAME = args.threshold_filename  # out/cocoruta_a=1_alpha=0.1_conf=frequency+model.pkl
            logger.info(f"Filename: {THRESHOLD_FILENAME}")
            filename = THRESHOLD_FILENAME.split("/")[-1].replace(".pkl", "")
            dataset_prefix = filename.split("_")[0]
            a = float(filename.split("_")[1].split("=")[1])
            alpha = float(filename.split("_")[2].split("=")[1])
            confidence_method = filename.split("_")[3].split("=")[1]
            logger.info("Metadata:")
            logger.info(f"dataset_prefix = {dataset_prefix}")
            logger.info(f"a = {a}")
            logger.info(f"alpha = {alpha}")
            logger.info(f"confidence_method = {confidence_method}")
        except Exception as e:
            raise ValueError(f"Option --threshold_filename must be passed when action is CONTROL (with command: python src/main.py --action CONTROL --threshold_filename [THRESHOLD_FILENAME]. Exception: {e}")
        cf: FactualConformalizer = joblib.load(THRESHOLD_FILENAME)
        logger.info(f"Successfully read the file {THRESHOLD_FILENAME}!")
        threshold = cf.get_threshold()
        logger.info(f"Threshold = {threshold}")
        logger.info("")

        logger.info("======== Reading the control dataset (inputs to control the output with CF) ========")
        try:
            CONTROL_FILENAME = args.control_filename  # data/cocoruta_control.jsonl
            logger.info(f"Filename: {CONTROL_FILENAME}")
            dataset_prefix = CONTROL_FILENAME.split("/")[-1].split("_")[0]
        except Exception as e:
            raise ValueError(f"Option --control_filename must be passed when action is CONTROL (with command: python src/main.py --action CONTROL --control_filename [CONTROL_FILENAME]. Exception: {e}")
        control_data = file_manager.load_output_from_file(CONTROL_FILENAME)
        n = len(control_data)
        logger.info(f"Successfully read the file {CONTROL_FILENAME} with {n} entries!")
        file_manager.head(control_data, n=5)  # shows the 5 first examples
        logger.info("")

        logger.info("======== Assigning for each subclaim factuality w.r.t. entailment ========")
        for entry in tqdm(control_data):
            for subclaim in entry["claims"]:
                subclaim["factual"] = int(subclaim["annotation"] in CORRECT_ANNOTATIONS)
        logger.info("")

        logger.info("======== Preparing for metrics computation ========")
        prepare_metrics(dataset=control_data, cf=cf, sm=sm)
        logger.info("")

        logger.info("======== Histogram of removed subclaims ========")
        cf_plt = FactualConformalizerPlotter(dataset_prefix)
        hist_filename = f"out/{dataset_prefix}_hist_a={a}_alpha={alpha}_conf={confidence_method}_controled.png"
        cf_plt.plot_hist(
            threshold=threshold,
            a=a,
            alpha=alpha,
            confidence_method=confidence_method,
            data=control_data,
            hist_filename=hist_filename
        )
        logger.info("")

        logger.info("======== Computing metrics of conditional CF ========")
        aggregated_metrics = compute_metrics(data=control_data, metrics=METRICS)
        logger.info("")

        logger.info("======== Generating file ========")
        jsonl_output = {"data": control_data} | aggregated_metrics
        file_manager.save_output_to_file(jsonl_output, f"out/{dataset_prefix}_controled.jsonl")
        logger.info(f"Successfully saved file out/{dataset_prefix}_controled.jsonl.")
        logger.info("")

    ############################################################################################
    ############################################################################################
    ############################################################################################


if __name__ == "__main__":
    setup_logger()
    log_env_config()
    main()