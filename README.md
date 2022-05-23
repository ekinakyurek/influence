# Tracing Knowledge in Language Models Back to the Training Data


## Setup local environment
```SHELL
python3 -m venv trex
source trex/bin/activate
pip install --upgrade pip
pip install -r requirements.txt # pip install -r requirements_gpu.txt
pre-commit install
```

## Run scripts

Set python path to the project root and then run a script
```SHELL
export PYTHONPATH=$(pwd)
bash scripts/bm25_results.sh
bash scripts/reranker.sh
```
