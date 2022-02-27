import json
import asyncio
import torch  # TODO(ekina): make this jax
from absl import app
from absl import flags
import numpy as np
from tqdm import tqdm

from src.tf_utils import tf
from src.tf_utils import get_tfexample_decoder_old
from src.tf_utils import get_index_decoder
from src.linalg_utils import normalize_segments


FLAGS = flags.FLAGS


flags.DEFINE_string('abstract_vectors', default=None,
                    help='input file path to convert')

flags.DEFINE_string('test_vectors', default=None,
                    help='input file path to convert')

flags.DEFINE_string('output_file', default=None,
                    help='output file path to writer neighbours')

flags.DEFINE_integer('gpu_workers', default=4,
                     help='number of gpus to distribute')

flags.DEFINE_integer('feature_size', default=4*2048,
                     help='size of vectors')

flags.DEFINE_integer('batch_size', default=10,
                     help='batch size to process at once')

flags.DEFINE_integer('topk', default=100,
                     help='batch size to process at once')

flags.DEFINE_integer('global_offset', default=0,
                     help='global offset of current vectors')

flags.DEFINE_integer('fake_data_size', default=None,
                     help='fake data size to test this code')

flags.DEFINE_bool('normalize', default=False,
                  help="normalize embeddings")


def load_a_shard_from_tfrecord(dataset,
                               feature_length,
                               positions,
                               normalize=False):
    """Loads one shard of a dataset from the dataset file."""
    shard_length = positions[1] - positions[0]
    ds_loader = dataset.skip(positions[0])\
                       .take(shard_length)\
                       .map(get_tfexample_decoder_old(feature_length))\
                       .batch(shard_length, num_parallel_calls=4)\
                       .as_numpy_iterator()

    X = next(iter(ds_loader))
    X = torch.from_numpy(X)
    if normalize:
        X = normalize_segments(X)
    return X


def load_a_shard_from_numpy(np_file,
                            feature_length,
                            positions,
                            normalize=False):
    """Loads one shard of a dataset from the dataset file."""
    X = np.load(np_file, mmap_mode='r')
    X = torch.from_numpy(X[positions[0]:positions[1], :])
    if normalize:
        X = normalize_segments(X)
    return X


def get_shard_positions(dataset, n_workers):
    """Gets start and end indicies of a shard in the merged dataset."""
    if type(dataset) == str:
        total_abstracts = FLAGS.fake_data_size
    else:
        total_abstracts = sum(1 for _ in dataset)
    shard_length = total_abstracts // n_workers
    positions = []
    start = 0
    for i in range(n_workers):
        current_start = start
        if i == n_workers - 1:
            current_end = total_abstracts
        else:
            current_end = current_start + shard_length
        positions.append((current_start, current_end))
        start = current_end
    return positions


def create_dummy_numpy_dataset(output_file, n, feature_size):
    """Creates dummy data to test this code."""
    # Somewhat this is extremely slow!
    X = np.random.rand(n, feature_size).astype(np.float32)
    np.save(output_file, X, allow_pickle=False)


def sharded_open(filename):
    if '@' in filename:
        prefix, no_shards = filename.split('@')
        no_shards = int(no_shards)
        filenames = [f'{prefix}-{str(i).zfill(5)}-of-{str(no_shards).zfill(5)}' for i in range(no_shards)]
        dataset = tf.data.TFRecordDataset(filenames=filenames)
    else:
        dataset = tf.data.TFRecordDataset(filename)
    return dataset


async def _prepare_shards(args):
    """Reads shards from the merged file and sends them to GPUs."""

    #  dataset = args.abstract_vectors
    # Create a dictionary describing the features.
    dataset = sharded_open(args.abstract_vectors)
    async def get_a_shard(i, index):
        shard = load_a_shard_from_tfrecord(dataset, args.feature_size, index, normalize=FLAGS.normalize).cuda(i, non_blocking=True)

        return shard

    positions = get_shard_positions(dataset, args.gpu_workers)

    loaders = [get_a_shard(*s) for s in enumerate(positions)]

    shards = await asyncio.gather(*loaders)

    return shards, positions


async def _single_shard_process(shard, batch, k):
    """Returns topk instances and scores for a given batch in a single shard"""
    scores = (shard @ batch.T.to(shard.device)).cpu()  # N x B
    return torch.topk(scores, k, dim=0)  # k x B


async def nn_retrieve(shards, batch, positions, k=100):
    """Returns topk instances and scores for a given batch."""
    retrievers = [_single_shard_process(shard, batch, k) for shard in shards]
    results = await asyncio.gather(*retrievers)

    value_lists, index_lists = zip(*results)

    temp = (index_lists[i] + positions[i][0] for i in range(len(index_lists)))
    index_tensor = torch.cat(tuple(temp), axis=0)
    value_tensor = torch.cat(value_lists, axis=0)
    scores, local_ids = torch.topk(value_tensor, k, dim=0)
    indices = index_tensor[local_ids, torch.arange(index_tensor.shape[-1])]
    return scores, indices


def get_indices(dataset, feature_length):
    ds = dataset.map(get_index_decoder(feature_length)).as_numpy_iterator()
    return np.array([d[0] for d in ds])


def main(_):
    # create_dummy_numpy_dataset(FLAGS.abstract_vectors,
    #                            FLAGS.fake_data_size,
    #                            FLAGS.feature_size)

    shards, positions = asyncio.run(_prepare_shards(FLAGS))
    print("reading abstract vectors from: ", FLAGS.abstract_vectors)
    abstract_indices = get_indices(sharded_open(FLAGS.abstract_vectors), FLAGS.feature_size)

    # create_dummy_tfrecord_dataset(FLAGS.test_vectors, 100, FLAGS.feature_size)
    test_indices = get_indices(sharded_open(FLAGS.test_vectors), FLAGS.feature_size)
    test_dataset = sharded_open(FLAGS.test_vectors)

    test_loader = test_dataset.map(get_tfexample_decoder_old(FLAGS.feature_size))\
                              .batch(FLAGS.batch_size)\
                              .as_numpy_iterator()

    results = []
    for batch in tqdm(test_loader):
        batch = torch.from_numpy(batch)

        if FLAGS.normalize:
            batch = normalize_segments(batch)

        scores, idxs = asyncio.run(nn_retrieve(shards,
                                               batch,
                                               positions,
                                               k=FLAGS.topk))
        # idxs = idxs + FLAGS.global_offset
        for i in range(scores.shape[-1]):
            line = {'scores': scores[:, i].tolist(),
                    'neighbor_ids': abstract_indices[idxs[:, i]].tolist()}
            results.append(line)

    results = [results[i] for i in np.argsort(test_indices)]

    with open(FLAGS.output_file, "w") as f:
        for res in results:
            print(json.dumps(res), file=f)


if __name__ == '__main__':
    app.run(main)
