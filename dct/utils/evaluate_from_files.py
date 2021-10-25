import pathlib
from dct.evaluation.sim.SIM import SIM
from dct.infer.cola_roberta_infer import ColaRobertaInfer
import numpy as np
import fasttext
import torch
from pytorch_lightning.metrics import Accuracy
from rich.table import Table
from itertools import takewhile


def evaluate_from_files(from_file, to_file, clf_ft_model_path, sim_model_file,
                        sim_sentencepiece_model_file,
                        cola_roberta_checkpoints_dir, cola_roberta_json_file):

    from_file = pathlib.Path(from_file)
    to_file = pathlib.Path(to_file)
    clf_ft_model_path = pathlib.Path(clf_ft_model_path)
    sim_model_file = pathlib.Path(sim_model_file)
    sim_sentencepiece_model_file = pathlib.Path(sim_sentencepiece_model_file)
    cola_roberta_checkpoints_dir = pathlib.Path(cola_roberta_checkpoints_dir)
    cola_roberta_json_file = pathlib.Path(cola_roberta_json_file)

    semantic_sim = SIM(
        str(sim_model_file), str(sim_sentencepiece_model_file), run_on_cpu=True
    )

    cola_roberta_infer = ColaRobertaInfer(
        checkpoints_dir=cola_roberta_checkpoints_dir,
        hparams_file=cola_roberta_json_file
    )

    dom_clf_model = fasttext.load_model(str(clf_ft_model_path))
    ft_style_mapping = {"__label__1": 0, "__label__2": 1}

    dev_transfer_accuracy = Accuracy()

    src_dom_sentences = []
    trg_dom_sentences = []
    with open(str(from_file), "r") as fp:
        for line in fp:
            line = line.strip().split()
            line = takewhile(lambda word: word != "<eos>", line)
            line = " ".join(list(line))
            src_dom_sentences.append(line.strip())

    with open(str(to_file), "r") as fp:
        for line in fp:
            line = line.strip().split()
            line = takewhile(lambda word: word != "<eos>", line)
            line = " ".join(list(line))
            trg_dom_sentences.append(line.strip())

    # calculate semantic similarity
    semantic_sim_scores = semantic_sim.find_similarity(src_dom_sentences, trg_dom_sentences)
    avg_semantic_sim_score = np.mean(semantic_sim_scores)
    avg_semantic_sim_score = round(avg_semantic_sim_score * 100, 2)

    # calculate fluency
    acceptable_preds = cola_roberta_infer.predict(trg_dom_sentences).tolist()
    percent_acceptable = np.mean(acceptable_preds)
    percent_acceptable = round(percent_acceptable * 100, 2)

    # transfer accuracy
    domain_predictions = dom_clf_model.predict(trg_dom_sentences)[0]
    # For every line the prediction can contain multiple predictions
    # Since ours is single prediction per line, take the first element
    domain_predictions = list(map(lambda pred: pred[0], domain_predictions))
    domain_predictions = [
        ft_style_mapping[pred] for pred in domain_predictions
    ]
    domain_predictions = torch.LongTensor(domain_predictions)
    true_labels = torch.LongTensor([1] * len(domain_predictions))
    acc = dev_transfer_accuracy(domain_predictions, true_labels)
    acc = round(acc.item() * 100, 2)

    aggregate = torch.mul(
        domain_predictions.float(), torch.Tensor(acceptable_preds)
    )
    aggregate = torch.mul(aggregate, torch.Tensor(semantic_sim_scores))
    test_aggregate = torch.mean(aggregate)
    test_aggregate = round(test_aggregate.item() * 100, 2)

    table = Table(title="Transfer Metrics")

    table.add_column("Metric", justify="right")
    table.add_column("#", justify="right")

    table.add_row("ACC", str(acc))
    table.add_row("FL", str(percent_acceptable))
    table.add_row("SIM", str(avg_semantic_sim_score))
    table.add_row("AGG", str(test_aggregate))

    return {
        "acc": acc,
        "fl": percent_acceptable,
        "sim": avg_semantic_sim_score,
        "agg": test_aggregate
    }


# if __name__ == "__main__":
#     from_file = "/home/rkashyap/abhi/arae/yelp/political_output/25_output_decoder_2_from.txt"
#     to_file = "/home/rkashyap/abhi/arae/yelp/political_output/25_output_decoder_2_tran.txt"
#     clf_ft_model_path = "/home/rkashyap/abhi/synarae/ftmodels/political_transfer_domclf.model"
#     sim_model_file = "/home/rkashyap/abhi/synarae/similarity_models/sim.pt"
#     sim_sentencepiece_model_file = "/home/rkashyap/abhi/synarae/similarity_models/sim.sp.30k.model"
#     cola_roberta_checkpoints_dir = "/home/rkashyap/abhi/synarae/experiments/cola_distilroberta-base_25e/checkpoints"
#     cola_roberta_json_file = "/home/rkashyap/abhi/synarae/experiments/cola_distilroberta-base_25e/hparams.json"
#
#     print(evaluate_from_files(
#         from_file=from_file,
#         to_file=to_file,
#         clf_ft_model_path=clf_ft_model_path,
#         sim_model_file=sim_model_file,
#         sim_sentencepiece_model_file=sim_sentencepiece_model_file,
#         cola_roberta_checkpoints_dir=cola_roberta_checkpoints_dir,
#         cola_roberta_json_file=cola_roberta_json_file
#     ))