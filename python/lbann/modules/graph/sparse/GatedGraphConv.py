import lbann 
from lbann.modules import Module 
from lbann.util import str_list
from lbann.modules.graph.utils import GraphVertexData
import lbann.modules
import math 

class GatedGraphConv(Module):
    """Gated Graph Convolution layer. For kernel details, see: 

    https://arxiv.org/abs/1511.05493

    Implementation in the spirit of:

    https://github.com/rusty1s/pytorch_geometric/blob/\
    master/torch_geometric/nn/conv/gated_graph_conv.py 
    """
    global_count = 0
    def __init__(self, 
                 output_channels,
                 num_layers = 1,
                 name = None):
        """Initialize GatedGraph layer
        Args: 
            output_channels (int): The output size of the node features 
            num_layers (int): Number of passes through the GRU (default: 1) 
            name (str): Name of the layers and prefix to use for the layers. 
            data_layout (str): Data layout (default: data parallel)  
        """
        super().__init__()

        ## Add Name for the components for the layer 
        GatedGraphConv.global_count +=1
        self.name = (name 
                    if name 
                    else 'GatedGraphConv_{}'.format(GatedGraphConv.global_count))


        ## Add variables
        self.output_channels = output_channels
        self.rnn  = lbann.modules.GRU(output_channels)

        self.num_layers = num_layers
        self.data_layout = data_layout

        self.weights = [] 

        for i in range(num_layers):
            
            weight_init = lbann.Weights(initializer = lbann.UniformInitializer(min =-1/(math.sqrt(output_channels)), 
                                                                               max = 1/(math.sqrt(output_channels))))
            weight_layer = lbann.WeightsLayer(dims = str_list([output_channels, output_channels]),
                                              weights = weight_init, 
                                              name = self.name+'_'+str(i)+'_weight',
                                              data_layout = self.data_layout)
            self.weights.append(weight_layer)
        

    def forward(self, X, A):
        """Call the GatedGraphConv
        Args:
            X (GraphVertexData): LBANN Data object, which is a collection of Layers. Each Layer is of
                                 the shape (1,input_channels) 
            A (Layer): Adjacency matrix input with shape (num_nodes, num_nodes)
        Returns: 
            LBANN_Data_Mat: The output after Gated Graph Kernel. 
                        The output can passed into another Graph Conv layer directly

        """

        input_features = X.size(1)
        num_nodes = X.size(0)

        if (input_features < self.output_channels):
            for i in range(num_nodes):
                num_zeros = self.output_channels - input_features 
                zeros = lbann.Constant(value = 0, num_neurons = str_list([1,num_zeros]), name = self.name+'_zero_'+str(i))
                X[i] = lbann.Concatenation(X[i], zeros, axis = 1)       
        elif (input_features > self.output_channels):
            ValueError('The feature size of the nodes {} cannot be greater than the output dimension {}'.
                        format(input_features, self.output_channels))
        
        X.update_num_features(self.output_channels)

        for layer in range(self.num_layers): 
            ##
            X_mat = X.get_mat()
            messages = lbann.MatMul(X_mat, self.weights[layer]) 
            aggregate = lbann.MatMul(A,messages)

            M = GraphVertexData.matrix_to_graph(aggregate, num_nodes, self.output_channels)

            for i in range(num_nodes):
                X[i] = lbann.Reshape(X[i], dims = str(self.output_channels))
                X[i] = lbann.Reshape(self.rnn(M[i], X[i])[1],
                                        dims = str_list([1, self.output_channels]))
        
        return X
