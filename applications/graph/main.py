import argparse
import os.path

import lbann
import lbann.contrib.lc.launcher
import lbann.contrib.args

import dataset
from utils import make_iterable, str_list
import utils.snap

# ----------------------------------
# Options
# ----------------------------------

# Command-line arguments
parser = argparse.ArgumentParser()
lbann.contrib.args.add_scheduler_arguments(parser)
parser.add_argument(
    '--job-name', action='store', default='lbann_node2vec', type=str,
    help='job name', metavar='NAME')
parser.add_argument(
    '--mini-batch-size', action='store', default=256, type=int,
    help='mini-batch size (default: 256)', metavar='NUM')
parser.add_argument(
    '--num-epochs', action='store', default=20, type=int,
    help='number of epochs (default: 20)', metavar='NUM')
parser.add_argument(
    '--latent-dim', action='store', default=128, type=int,
    help='latent space dimensions (default: 128)', metavar='NUM')
args = parser.parse_args()

# ----------------------------------
# Embedding weights
# ----------------------------------

embeddings = lbann.Weights(initializer=lbann.NormalInitializer(mean=0,
                                                               standard_deviation=1),
                           name='embeddings')

# ----------------------------------
# Construct layer graph
# ----------------------------------

# Properties of graph and random walk
num_graph_nodes = dataset.max_graph_node_id() + 1
walk_length = dataset.sample_dims()[0]

# Input is a sequence of graph node IDs
input_ = lbann.Identity(lbann.Input(), device='cpu')
input_slice = lbann.Slice(input_,
                          slice_points=str_list(range(walk_length+1)),
                          device='cpu')
walk = []
for _ in range(walk_length):
    walk.append(lbann.Identity(input_slice, device='cpu'))

# Skip-gram architecture
latent = lbann.Embedding(walk[0],
                         weights=embeddings,
                         num_embeddings=num_graph_nodes,
                         embedding_dim=args.latent_dim,
                         device='cpu')
pred = lbann.FullyConnected(latent,
                            weights=embeddings,
                            num_neurons=num_graph_nodes,
                            has_bias=False,
                            transpose=True,
                            device='cpu')
pred = lbann.Softmax(pred, device='cpu')

# Objective function
ground_truth = lbann.Sum([lbann.OneHot(node, size=num_graph_nodes, device='cpu')
                          for node in walk[1:]],
                         device='cpu')
obj = lbann.CrossEntropy([pred, ground_truth], device='cpu')

# ----------------------------------
# Create data reader
# ----------------------------------

reader = lbann.reader_pb2.DataReader()
_reader = reader.reader.add()
_reader.name = 'python'
_reader.role = 'train'
_reader.percent_of_data_to_use = 1.0
_reader.python.module = 'dataset'
_reader.python.module_dir = os.path.dirname(os.path.realpath(__file__))
_reader.python.sample_function = 'get_sample'
_reader.python.num_samples_function = 'num_samples'
_reader.python.sample_dims_function = 'sample_dims'

# ----------------------------------
# Run LBANN
# ----------------------------------

# Create LBANN objects
trainer = lbann.Trainer()
model = lbann.Model(args.mini_batch_size,
                    args.num_epochs,
                    layers=lbann.traverse_layer_graph(input_),
                    objective_function=obj,
                    callbacks=[lbann.CallbackPrint(),
                               lbann.CallbackTimer()])
opt = lbann.SGD(learn_rate=0.01, momentum=0.9)

# Run LBANN
kwargs = lbann.contrib.args.get_scheduler_kwargs(args)
lbann.contrib.lc.launcher.run(trainer, model, reader, opt,
                              job_name=args.job_name,
                              **kwargs)
