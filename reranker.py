import os
import pathlib
import subprocess
import time
from absl import app, flags, logging


FLAGS = flags.FLAGS

flags.DEFINE_string(
    "metrics_file",
    default=None,
    help=(
        "input metrics file that stores baseline statistics and (examples, nn"
        " abstracts)"
    ),
)

flags.DEFINE_string(
    "baseline_metrics_file",
    default=None,
    help="output file for experiment results",
)

flags.DEFINE_string(
    "baseline_nn_file", default=None, help="nn for baseline file"
)

flags.DEFINE_string(
    "checkpoint_folders",
    default=None,
    help="last checkpoint of the model to evaluate",
)

flags.DEFINE_integer(
    "beam_size", default=3, help="beam size for accuracy calculations"
)

flags.DEFINE_integer("seed", default=10, help="seed")

flags.DEFINE_float(
    "baseline_reweight",
    default=-1,
    help="ensemble with reweighted baseline scores",
)

flags.DEFINE_string("data_root", default="LAMA/data/", help="data folder")

flags.DEFINE_string(
    "lama_folder",
    default="LAMA/data/TREx_lama_templates_v3",
    help="lama data folder name; should be inside data folder",
)

flags.DEFINE_string(
    "exp_folder",
    default="LAMA/data/metrics/reranker/unfiltered",
    help="name for exp folder under data root",
)

flags.DEFINE_string("gpus_to_use", default=None, help="coma seperated gpu ids")


def wait_for_files(files):
    logging.info("waiting for files")
    logging.info(repr(files))
    while True:
        all_files = True
        for file in files:
            if not os.path.isfile(file):
                all_files = False
        if all_files:
            break
        time.sleep(60)


def assign_to_gpu(gpus, file):
    logging.info(f"waiting for empty gpu for {file}")
    while True:
        for (k, v) in gpus.items():
            if len(v) == 0:
                v.append(file)
                return k
            for i in range(len(v)):
                if os.path.isfile(v[i]) or os.path.isdir(v[i]):
                    del v[i]
                    v.append(file)
                    return k
        time.sleep(60)


def main(_):
    uri_file = os.path.join(FLAGS.lama_folder, "abstracts", "all_used_uris.txt")
    hashmap_file = os.path.join(
        FLAGS.lama_folder, "abstracts", "hashmap_used.json"
    )
    test_file = os.path.join(FLAGS.lama_folder, "all.tfrecord")
    abstract_file = os.path.join(
        FLAGS.lama_folder, "abstracts", "all_used.jsonl"
    )
    checkpoint_folders = FLAGS.checkpoint_folders.split(",")

    gpus = list(map(int, FLAGS.gpus_to_use.split(",")))
    gpus = {id: [] for id in gpus}
    print(f"gpus: {gpus}")

    evaluate_cmd = (
        "export PYTHONHASHSEED=0;"
        "python -u eval/evaluate.py "
        f"--abstract_uri_list {uri_file} "
        f"--abstract_file {abstract_file} "
        f"--test_data {test_file} "
        f"--hashmap_file {hashmap_file} "
        f"--nn_list_file {FLAGS.baseline_nn_file} "
        "--disable_tqdm "
        f"--output_file {FLAGS.baseline_metrics_file};"
        "deactivate"
    )

    logging.info(
        "Running baseline evaluations..."
        f"Metrics will be outputted to {FLAGS.baseline_metrics_file}"
    )

    subprocess.run(evaluate_cmd, shell=True, check=True)

    header_cmd = (
        'eval "$(conda shell.bash hook)";conda activate transformers;export'
        " PYTHONHASHSEED=0;"
    )

    for i in range(3):
        output_metric_folder = os.path.join(FLAGS.exp_folder, f"seed_{i}")
        for subset in ("learned", "random"):
            os.makedirs(output_metric_folder, exist_ok=True)

            baseline_prefix = os.path.join(output_metric_folder, f"{subset}/")
            os.makedirs(baseline_prefix, exist_ok=True)
            baseline_eval_file = os.path.join(baseline_prefix, "eval_detailed")

            gpu = assign_to_gpu(gpus, f"{baseline_eval_file}.pickle")
            gpu_header = f"export CUDA_VISIBLE_DEVICES={gpu};"

            pre_params = (
                f"--metrics_file={FLAGS.baseline_metrics_file} "
                f"--seed={i} "
                f"--checkpoint_folders={FLAGS.checkpoint_folders} "
                f"--output_metrics_prefix={baseline_eval_file} "
                "--gpu=0 "
                "--disable_tqdm "
            )

            if subset == "corrects":
                pre_params += "--only_correct "
            elif subset == "wrongs":
                pre_params += "--only_wrongs "
            else:
                pre_params += "--only_learned "

            baseline_log_prefix = os.path.join(baseline_prefix, "logs/")
            os.makedirs(baseline_log_prefix, exist_ok=True)

            logging.info(
                f"Experiment files {baseline_log_prefix}\nParams:\n{pre_params}"
            )

            pre_cmd = (
                f"python -u eval/reranker_pre.py {pre_params} > "
                f"{baseline_log_prefix}/pre.log 2>"
                f"{baseline_log_prefix}/pre.err;"
            )

            logging.info(f"RUN: {pre_cmd}")
            subprocess.run(gpu_header + header_cmd + pre_cmd, shell=True)

            for eos in ("no_eos",):
                for accum in ("accum", "no_accum"):

                    ckpt_prefix = os.path.join(
                        baseline_prefix, f"{eos}_{accum}/"
                    )
                    ckpt_log_prefix = os.path.join(ckpt_prefix, "logs/")

                    os.makedirs(ckpt_log_prefix, exist_ok=True)
                    ckpt_scores_prefix = os.path.join(ckpt_prefix, "scores/")

                    os.makedirs(ckpt_scores_prefix, exist_ok=True)

                    files_to_check = []

                    for c, folder in enumerate(checkpoint_folders):

                        checkpoint_name = pathlib.PurePath(folder).name

                        output_ckpt_file = os.path.join(
                            ckpt_scores_prefix, f"{checkpoint_name}.pickle"
                        )

                        # files_to_check.append(output_ckpt_file)
                        gpu = assign_to_gpu(gpus, output_ckpt_file)

                        gpu_header = f"export CUDA_VISIBLE_DEVICES={gpu};"

                        ckpt_params = (
                            f"--metrics_file={baseline_eval_file}.pickle "
                            f"--seed={i} "
                            f"--checkpoint_folder={folder} "
                            f"--output_metrics_prefix={ckpt_scores_prefix} "
                            "--gpu=0 "
                            "--disable_tqdm "
                        )

                        if eos == "eos":
                            ckpt_params += "--include_eos "
                        if accum == "accum":
                            ckpt_params += "--load_accums "

                        if c == len(checkpoint_folders) - 1:
                            ckpt_params += "--calculate_activation_scores"
                        else:
                            ckpt_params += "--calculate_gradient_scores"

                        ckpt_cmd = (
                            "python -u eval/reranker_single_checkpoint.py"
                            f" {ckpt_params} >{ckpt_log_prefix}/ckpt.{c}.log;"
                        )

                        logging.info(f"RUN: {ckpt_cmd}")

                        subprocess.Popen(
                            gpu_header + header_cmd + ckpt_cmd, shell=True
                        )
                        time.sleep(10)

                    wait_for_files(files_to_check)
                    # time.sleep(600)  # give some time for process for finishing the writing to file

                    post_params = (
                        f"--metrics_file={baseline_eval_file}.pickle "
                        f"--seed={i} "
                        f"--scores_folder={ckpt_scores_prefix} "
                        "--exp_type=layers "
                        f"--output_metrics_file={ckpt_prefix}/results_detailed "
                        "--disable_tqdm "
                    )

                    post_cmd = (
                        f"python -u eval/reranker_post.py {post_params} >"
                        f"{ckpt_log_prefix}/post.log 2> "
                        f"{ckpt_log_prefix}/post.err;"
                    )

                    logging.info(f"RUN: {post_cmd}")

                    subprocess.Popen(header_cmd + post_cmd, shell=True)


if __name__ == "__main__":
    app.run(main)
