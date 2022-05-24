# Tracing Knowledge in Language Models Back to the Training Data

Paper: [Tracing Knowledge in Language Models Back to the Training Data](https://arxiv.org/abs/2205.11482)   
Ekin Akyürek, Tolga Bolukbasi, Frederick Liu, Binbin Xiong, Ian Tenney, Jacob Andreas, Kelvin Guu (2022)

## Setup local environment
```SHELL
python3 -m venv trex
source trex/bin/activate
pip install --upgrade pip
pip install -r requirements.txt # pip install -r requirements_gpu.txt
pre-commit install
```

## Data & Benchmark
Please see the detailed information about data in https://huggingface.co/datasets/ekinakyurek/ftrace before using it.

## Run scripts

Set python path to the project root and then run a script
```SHELL
export PYTHONPATH=$(pwd)
bash scripts/bm25_results.sh
bash scripts/reranker.sh
```


## Citation
```
@misc{https://doi.org/10.48550/arxiv.2205.11482,
  doi = {10.48550/ARXIV.2205.11482},
  url = {https://arxiv.org/abs/2205.11482},
  author = {Akyürek, Ekin and Bolukbasi, Tolga and Liu, Frederick and Xiong, Binbin and Tenney, Ian and Andreas, Jacob and Guu, Kelvin},
  keywords = {Computation and Language (cs.CL), Information Retrieval (cs.IR), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Tracing Knowledge in Language Models Back to the Training Data},
  publisher = {arXiv},
  year = {2022}, 
}
```