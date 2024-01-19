from dct.utils.evaluate_from_files import evaluate_from_files
from dct.utils.multinomial_label_acc import MultinomialLabelAccuracy
from rich.console import Console
from rich.table import Table
import pathlib
import csv
import click


@click.command()
@click.option("--from-file", type=str)
@click.option("--to-file", type=str)
@click.option("--clf-ft-model-path", type=str)
@click.option("--sim-model-file", type=str)
@click.option("--sim-sentencepiece-model-file", type=str)
@click.option("--cola-roberta-checkpoints-dir", type=str)
@click.option("--cola-roberta-json-file", type=str)
@click.option("--src-dom-saliency-file", type=str)
@click.option("--trg-dom-saliency-file", type=str)
@click.option("--model-name", type=str)
@click.option("--gen-method", type=str)
@click.option("--data-name", type=str)
@click.option("--main-results-csv-filename", type=str)
@click.option("--constrain-results-csv-filename", type=str)
def write_results_table(
    from_file,
    to_file,
    clf_ft_model_path,
    sim_model_file,
    sim_sentencepiece_model_file,
    cola_roberta_checkpoints_dir,
    cola_roberta_json_file,
    src_dom_saliency_file,
    trg_dom_saliency_file,
    model_name: str,
    gen_method: str,
    data_name,
    main_results_csv_filename,
    constrain_results_csv_filename,
):
    """Writes the results for both the constraint generation
    and the aggregate score

    Returns
    -------

    """

    ###################################
    # Create csv file for main results
    ##################################
    main_results_csv_filename = pathlib.Path(main_results_csv_filename)
    main_results_csv_fields = ["model", "data", "gen_method", "acc", "fl", "sim", "agg"]
    if not main_results_csv_filename.is_file():
        fp = open(main_results_csv_filename, "w")
        main_results_csv_writer = csv.DictWriter(fp, fieldnames=main_results_csv_fields)
        main_results_csv_writer.writeheader()

    else:
        fp = open(main_results_csv_filename, "a")
        main_results_csv_writer = csv.DictWriter(fp, fieldnames=main_results_csv_fields)

    ###################################
    # Create csv file for constrain result
    ##################################
    constrain_results_csv_filename = pathlib.Path(constrain_results_csv_filename)
    constrain_results_csv_fields = ["model", "property", "data", "gen_method", "f1"]
    if not constrain_results_csv_filename.is_file():
        fp = open(constrain_results_csv_filename, "w")
        constrain_results_csv_writer = csv.DictWriter(fp, fieldnames=constrain_results_csv_fields)
        constrain_results_csv_writer.writeheader()

    else:
        fp = open(constrain_results_csv_filename, "a")
        constrain_results_csv_writer = csv.DictWriter(fp, fieldnames=constrain_results_csv_fields)

    console = Console()

    results_dict = evaluate_from_files(
        from_file=from_file,
        to_file=to_file,
        clf_ft_model_path=clf_ft_model_path,
        sim_model_file=sim_model_file,
        sim_sentencepiece_model_file=sim_sentencepiece_model_file,
        cola_roberta_checkpoints_dir=cola_roberta_checkpoints_dir,
        cola_roberta_json_file=cola_roberta_json_file,
    )
    acc = results_dict["acc"]
    fl = results_dict["fl"]
    sim = results_dict["sim"]
    agg = results_dict["agg"]

    main_table = Table(title=f"{from_file}->{to_file}")
    main_table.add_column("MODEL")
    main_table.add_column("DATA")
    main_table.add_column("GENMETHOD")
    main_table.add_column("ACC")
    main_table.add_column("FL")
    main_table.add_column("SIM")
    main_table.add_column("AGG")
    main_table.add_row(model_name, data_name, gen_method, str(acc), str(fl), str(sim), str(agg))

    main_results_csv_writer.writerow(
        {
            "model": model_name,
            "data": data_name,
            "gen_method": gen_method,
            "acc": str(acc),
            "fl": str(fl),
            "sim": str(sim),
            "agg": str(agg),
        }
    )

    console.print(main_table)

    label_acc = MultinomialLabelAccuracy(
        pred_filename=to_file,
        true_filename=from_file,
        src_dom_saliency_file=src_dom_saliency_file,
        trg_dom_saliency_file=trg_dom_saliency_file,
        avg_length=10,
    )

    label_acc_results = label_acc.get_accuracy()

    constrain_table = Table(title=f"CONSTRAINTS - {from_file}->{to_file}")
    constrain_table.add_column("MODEL")
    constrain_table.add_column("DATA")
    constrain_table.add_column("GENMETHOD")
    constrain_table.add_column("PROPERTY")
    constrain_table.add_column("F1")

    for label_name in label_acc_results:
        fmeasure = str(label_acc_results[label_name]["f-score"])
        constrain_results_csv_writer.writerow(
            {
                "model": model_name,
                "property": label_name,
                "data": data_name,
                "gen_method": gen_method,
                "f1": fmeasure,
            }
        )
        constrain_table.add_row(model_name, data_name, gen_method, label_name, fmeasure)

    console.print(constrain_table)


if __name__ == "__main__":
    write_results_table()
