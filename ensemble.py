import os
import subprocess
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
    "fact_to_ids_file",
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

flags.DEFINE_string(
    "load_exp_folder",
    default=None,
    help="name for exp folder to load the splits from",
)

flags.DEFINE_string("gpus_to_use", default=None, help="coma seperated gpu ids")


def main(_):

    # checkpoint_folders = FLAGS.checkpoint_folders.split(",")

    gpus = list(map(int, FLAGS.gpus_to_use.split(",")))
    gpus = {id: [] for id in gpus}
    print(f"gpus: {gpus}")

    header_cmd = (
        'eval "$(conda shell.bash hook)";conda activate transformers;export'
        " PYTHONHASHSEED=0;"
    )

    for i in range(3):

        exp_folder = FLAGS.load_exp_folder

        output_metric_folder = os.path.join(exp_folder, f"seed_{i}")

        for subset in ("learned",):
            baseline_prefix = os.path.join(output_metric_folder, f"{subset}/")
            baseline_eval_file = os.path.join(baseline_prefix, "eval_detailed")

            for eos in ("no_eos",):
                for accum in ("accum",):

                    ckpt_prefix = os.path.join(
                        FLAGS.load_exp_folder,
                        f"seed_{i}",
                        subset,
                        f"{eos}_{accum}/",
                    )

                    ckpt_log_prefix = os.path.join(ckpt_prefix, "logs/")

                    ckpt_scores_prefix = os.path.join(ckpt_prefix, "scores/")

                    ckpt_prefix = os.path.join(
                        FLAGS.exp_folder,
                        f"seed_{i}",
                        subset,
                        f"{eos}_{accum}/",
                    )

                    post_params = (
                        f"--metrics_file={baseline_eval_file}.pickle "
                        f"--seed={i} "
                        f"--scores_folder={ckpt_scores_prefix} "
                        "--exp_type=layers "
                        f"--output_metrics_file={ckpt_prefix}/results_ensemble "
                        f"--alpha {FLAGS.baseline_reweight} "
                        "--reweight_type arithmetic "
                        "--disable_tqdm "
                    )

                    ckpt_log_prefix = os.path.join(ckpt_prefix, "logs/")

                    post_cmd = (
                        f"python -u eval/reranker_post.py {post_params} >"
                        f"{ckpt_log_prefix}/post.log 2> "
                        f"{ckpt_log_prefix}/post.err;"
                    )

                    logging.info(f"RUN: {post_cmd}")

                    subprocess.Popen(header_cmd + post_cmd, shell=True)


if __name__ == "__main__":
    app.run(main)
