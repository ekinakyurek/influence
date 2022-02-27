import functools
from typing import Optional, Sequence, Mapping
import torch  # TODO(ekina): make this jax
from absl import app
from absl import flags
import numpy as np
import eval.reranker
from eval.reranker import LayerConfig, collapse_abstracts_and_scores
from eval.reranker import get_gradients
from eval.reranker import get_activations
from eval.reranker import tokenize
from eval.reranker import merge_new_scores_to_dict
from eval.reranker import main
from src.linalg_utils import f_normalize


FLAGS = flags.FLAGS


def get_scores(vectors1, vectors2, f=lambda x: x):
    """Get dot product of dictionary of vectors with a preprocesser function f"""
    return {k: torch.dot(f(v), f(vectors2[k])).item() for k, v in vectors1.items()}


def get_all_scores_for_model(model, query, abstracts, encoder):
    """Get both cosine and uncosine scores for all the abstracts"""
    query_grad = encoder(model, query)
    scores = []
    nscores = []
    for i, abstract in enumerate(abstracts):
        abstract_grad = encoder(model, abstract)
        score = get_scores(query_grad, abstract_grad)
        scores.append(score)
        nscore = get_scores(query_grad, abstract_grad, f=f_normalize)
        nscores.append(nscore)
        del abstract_grad
    return scores, nscores


def merge_model_scores(scores):
    """Merge scores obtained from n checkpoints into single score by taking mean"""
    assert len(scores) > 0
    abstract_scores = []
    for j in range(len(scores[0])):
            abstract_scores.append(
                {k: np.mean([s[j][k] for s in scores])  # Mean over checkpoints
                                    for k, _ in scores[0][j].items()})
    return abstract_scores


def get_all_scores(models, tokenizer, query, abstracts):
    """Get both activation scores and mean gradient scores over checkpoints for list of models
       Note: We only take activation scores for last checkpoint.
    """
    query = tokenize(tokenizer, query)
    abstracts = [tokenize(tokenizer, a) for a in abstracts]
    all_scores = {'dot': {}, 'cosine': {}}
    
    for encoder in (get_gradients, get_activations):
        if encoder == get_activations:
            score, nscore = get_all_scores_for_model(models[-1], query, abstracts, encoder)
        else:
            scores = []
            nscores = []
            for model in models[:-1]:
                score, nscore = get_all_scores_for_model(model, query, abstracts, encoder)
                scores.append(score)
                nscores.append(nscore)
            score = merge_model_scores(scores)
            nscore = merge_model_scores(nscores)
       
        all_scores['dot'] = merge_new_scores_to_dict(all_scores['dot'], score)
        all_scores['cosine'] = merge_new_scores_to_dict(all_scores['cosine'], nscore)
   
    return all_scores
    

def rerank_with_scores(abstracts: Sequence[Mapping], layer_scores: Mapping, layers: Optional[LayerConfig] = None, collapse: bool = False, normalize: bool = False):
    """Given layers prefixes we sum scores of these layers and rerank the abstracts"""
    abstract_scores = []
    if layers is not None:
        # Assuming our layer configurations are prefix codes
        def findindex(key):
            return np.where([key.startswith(layer)
                             for layer in layers.layer_prefixes])[0]
        
        sum_names = [weight_name for weight_name in layer_scores[0].keys() if len(findindex(weight_name)) > 0]
        w_weights = {weight_name: layers.layer_weights[findindex(weight_name)[0]] for weight_name in sum_names}
    else:
        sum_names = list(layer_scores[0].keys())
        w_weights = {k: 1.0 for k in sum_names}

    for layer_score in layer_scores:
        abstract_scores.append(np.sum([layer_score[k] * w_weights[k] for k in sum_names]))

    scores = np.array(abstract_scores)
    # merge abstracts and scores here
    if collapse:
        scores, abstracts = collapse_abstracts_and_scores(scores, abstracts)
    
    sorted_idxs = np.argsort(-scores)
    abstracts_reranked = [abstracts[i] for i in sorted_idxs]
    scores_reranked = scores[sorted_idxs]
    
    return abstracts_reranked, scores_reranked


reranker_local_main = functools.partial(main, reranker_fn=rerank_with_scores)


if __name__ == '__main__':
    app.run(reranker_local_main)
    