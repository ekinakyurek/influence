# Few-shot and 2-hop Knowledge Probing for Pretrained Language Models

This repo contains code for https://arxiv.org/abs/2109.02772 .

This repo contains two major parts: (1) An extention dataset for 2-hop relation probing based on LAMA: TREx_multihop. (Sorry that the name "multihop" in the code could be misleading, it only contains 2-hop relations.) (2) Few-shot training.

This code is based on LAMA(https://github.com/facebookresearch/LAMA).

## Preparation

Please create a virtual environment with python 3.7, and do 
```
pip install -r requirements.txt
```

Then, go to the LAMA/ directory, and
```
source ../add_path.sh
```
This changes the PYTHONPATH variable, please change it to match your path.

Now let's prepare the TREx data-set. Goto LAMA/data, and 
```
wget https://dl.fbaipublicfiles.com/LAMA/data.zip
unzip data.zip
mv data/TREx .
rm -r data && rm data.zip
```
Note that the trex_multihop dataset is already in data/ .

A note on vocab: Instead of using a common vocab (as in LAMA), we directly use the vocab from the Roberta model.

Now we should be ready to go! For 0-fewshot baseline, just run:
```
python scripts/run_experiments.py --dataset trex_multihop
```
You can switch the dataset between trex_multihop and trex.

## Few-shot Training

Below, we give some example of commands that can be used to reproduce results in our report. 
If you have any questions, feel free to raise an issue.

### Prompt Engineering

Below is an example commmand to run with OptiPrompt(https://arxiv.org/abs/2104.05240):
```
python scripts/run_experiments.py --dataset trex_multihop --fewshot_ft 10 --fewshot_ft_tuneseed_num 20 --relation_mode relvec --fewshot_ft_param_mode only_relvec --relvec_learning_rate 0.025
```
"--fewshot_ft 10" means that we use 10 few-shot examples (the code will automaticallly add another 10 for validation) to train the bitfit bias. You can change it to 5 or 30.

The option of "fewshot_ft_tuneseed_num" means that we tune 20 different random seed (we find that this approach is sensitive to random seed), and select the checkpoint with best validation loss.
This approach is also sensitive to learning rate, we tune the relvec_learning_rate in {0.01, 0.025, 0.1, 0.5}. (Note that the best learning rate different from case by case)
You can add "--relvec_initialize_mode from_template" to use embeddings from manual template to initialize the vectors.

### In-context Learning

Below is an command to do in-context learning (which involves no parameter tuning).
```
python scripts/run_experiments.py --dataset trex_multihop --fewshot_context 10 --fewshot_ft_tuneseed_num 20 --template_source lama
```
Here, the "fewshot_ft_tuneseed_num" option means that we are tuning the random order of the few-shot examples in the context. The "--template_source lama" means that we are using manually created template, you can switch it to "default" templates.

### BitFit

Below is an example command to run with bitfit(https://arxiv.org/abs/2106.10199):
```
python scripts/run_experiments.py --dataset trex_multihop --template_source lama --fewshot_ft 10 --learning_rate 0.01 --fewshot_ft_param_mode bitfit --fewshot_ft_tuneseed_num 20
```
This means that we use 10 few-shot examples (the code will automaticallly add another 10 for validation) to train the bitfit bias, the option of "fewshot_ft_tuneseed_num" means that we tune 20 different random seed, and select the checkpoint with best validation loss. The "--template_source lama" means that we are using manually created template, you can switch it to "default" templates.
For BitFit, we tune the learning rate with {0.01, 0.001, 0.0001}.

BitFit can also be combined with OptiPrompt:
```
python scripts/run_experiments.py --fewshot_ft 10 --dataset trex_multihop --fewshot_ft_param_all_lr_overwrite True --fewshot_ft_tuneseed_num 20 --relation_mode relvec --fewshot_ft_param_mode bitfit --learning_rate 0.01
```

## Citation

If you use our code, we would really appreciate if you could cite our work!
```
@misc{he2021empirical,
      title={An Empirical Study on Few-shot Knowledge Probing for Pretrained Language Models}, 
      author={Tianxing He and Kyunghyun Cho and James Glass},
      year={2021},
      eprint={2109.02772},
      archivePrefix={arXiv},
      primaryClass={cs.AI}
}
```
