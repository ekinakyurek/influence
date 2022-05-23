# Tracing Knowledge in Language Models Back to the Training Data

Paper: [Tracing Knowledge in Language Models Back to the Training Data]()   
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
Please see the information about data in https://huggingface.co/datasets/ekinakyurek/ftrace

## Run scripts

Set python path to the project root and then run a script
```SHELL
export PYTHONPATH=$(pwd)
bash scripts/bm25_results.sh
bash scripts/reranker.sh
```
