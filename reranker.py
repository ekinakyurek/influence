import os
import time
import subprocess
import pathlib
from absl import app
from absl import flags
from absl import logging

FLAGS = flags.FLAGS

flags.DEFINE_string('metrics_file', default=None,
                    help='input metrics file that stores baseline statistics and (examples, nn abstracts)')

flags.DEFINE_string('baseline_metrics_file', default=None,
                    help='output file for experiment results')

flags.DEFINE_string('baseline_nn_file', default=None,
                    help='nn for baseline file')

flags.DEFINE_string('checkpoint_folders', default=None,
                    help='last checkpoint of the model to evaluate')

flags.DEFINE_integer('beam_size', default=3,
                     help="beam size for accuracy calculations")

flags.DEFINE_integer('seed', default=10, help="seed")

flags.DEFINE_string('data_root', default="LAMA/data/",
                    help="data folder")

flags.DEFINE_string('lama_folder', default="LAMA/data/TREx_lama_templates_v3",
                    help="lama data folder name; should be inside data folder")

flags.DEFINE_string("exp_folder", default="LAMA/data/metrics/reranker/unfiltered",
                    help="name for exp folder under data root")


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


def main(_):
    uri_file = os.path.join(FLAGS.lama_folder, 'abstracts', 'all_used_uris.txt')
    hashmap_file = os.path.join(FLAGS.lama_folder, 'abstracts', 'hashmap_used.json')
    test_file = os.path.join(FLAGS.lama_folder, 'all.tfrecord')
    abstract_file = os.path.join(FLAGS.lama_folder, 'abstracts', 'all_used.jsonl')
    checkpoint_folders = FLAGS.checkpoint_folders.split(',')

    evaluate_cmd = (f"export PYTHONHASHSEED=0;"
                    f"python -u eval/evaluate.py "
                    f"--abstract_uri_list {uri_file} "
                    f"--abstract_file {abstract_file} "
                    f"--test_data {test_file} "
                    f"--hashmap_file {hashmap_file} "
                    f"--nn_list_file {FLAGS.baseline_nn_file} "
                    f"--output_file {FLAGS.baseline_metrics_file};"
                    f"deactivate")

    logging.info(f"Running baseline evaluations..."
                 f"Metrics will be outputted to {FLAGS.baseline_metrics_file}")

    # subprocess.run(evaluate_cmd, shell=True, check=True)

    header_cmd = ("eval \"$(conda shell.bash hook)\";"
                  "conda activate transformers;"
                  "export PYTHONHASHSEED=0;")

    for i in range(1):
        output_metric_folder = os.path.join(FLAGS.exp_folder, f"seed_{i}")
        for subset in ("learned", ):
            os.makedirs(output_metric_folder, exist_ok=True)

            baseline_prefix = os.path.join(output_metric_folder,
                                           f"{subset}/")
            os.makedirs(baseline_prefix, exist_ok=True)
            baseline_eval_file = os.path.join(baseline_prefix, "eval")

            pre_params = (f"--metrics_file={FLAGS.baseline_metrics_file} "
                          f"--seed={i} "
                          f"--checkpoint_folders={FLAGS.checkpoint_folders} "
                          f"--output_metrics_prefix={baseline_eval_file} "
                          f"--gpu=0 ")

            if subset == "corrects":
                pre_params += "--only_correct "
            elif subset == "wrongs":
                pre_params += "--only_wrongs "
            else:
                pre_params += "--only_learned "

            baseline_log_prefix = os.path.join(baseline_prefix, "logs/")
            os.makedirs(baseline_log_prefix, exist_ok=True)

            logging.info(f"Experiment files {baseline_log_prefix}\n"
                         f"Params:\n{pre_params}")

            pre_cmd = (f"python -u eval/reranker_pre.py {pre_params} > "
                       f"{baseline_log_prefix}/pre.log")

            logging.info(f"RUN: {pre_cmd}")

            # subprocess.run(header_cmd + pre_cmd, shell=True)

            for eos in ("no_eos", ):
                for accum in ("accum", ):

                    ckpt_prefix = os.path.join(baseline_prefix,
                                                   f"{eos}_{accum}/")
                    ckpt_log_prefix = os.path.join(ckpt_prefix,
                                                       "logs/")

                    os.makedirs(ckpt_log_prefix, exist_ok=True)
                    ckpt_scores_prefix = os.path.join(ckpt_prefix,
                                                          "scores/")

                    os.makedirs(ckpt_scores_prefix, exist_ok=True)

                    files_to_check = []

                    for c, folder in enumerate(checkpoint_folders):

                        if c != len(checkpoint_folders) - 1:
                            continue

                        ckpt_params = (
                            f"--metrics_file={baseline_eval_file}.pickle "
                            f"--seed={i} "
                            f"--checkpoint_folder={folder} "
                            f"--output_metrics_prefix={ckpt_scores_prefix} "
                            f"--gpu={c} ")

                        checkpoint_name = pathlib.PurePath(folder).name

                        output_ckpt_file = os.path.join(
                                                ckpt_scores_prefix,
                                                f"{checkpoint_name}.pickle")

                        files_to_check.append(output_ckpt_file)

                        if eos == "eos":
                            ckpt_params += "--include_eos "
                        if accum == "accum":
                            ckpt_params += "--load_accums "

                        if c == len(checkpoint_folders) - 1:
                            ckpt_params += "--calculate_activation_scores"
                        else:
                            ckpt_params += "--calculate_gradient_scores"

                        ckpt_cmd = (f"python -u eval/reranker_single_checkpoint.py {ckpt_params} >"
                                    f"{ckpt_log_prefix}/ckpt.{c}.log;")

                        logging.info(f"RUN: {ckpt_cmd}")

                        # subprocess.Popen(header_cmd + ckpt_cmd, shell=True)
                        time.sleep(5)

                    wait_for_files(files_to_check)
                    time.sleep(5)

                    post_params = (
                            f"--metrics_file={baseline_eval_file}.pickle "
                            f"--seed={i} "
                            f"--scores_folder={ckpt_scores_prefix} "
                            f"--exp_type=layers "
                            f"--output_metrics_file={ckpt_prefix}/results "
                            )

                    post_cmd = (f"python -u eval/reranker_post.py {post_params} >"
                                f"{ckpt_log_prefix}/post.log;")

                    logging.info(f"RUN: {post_cmd}")

                    subprocess.Popen(header_cmd + post_cmd, shell=True)


if __name__ == '__main__':
    app.run(main)