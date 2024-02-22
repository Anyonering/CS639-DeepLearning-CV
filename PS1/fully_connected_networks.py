"""
Implements fully connected networks in PyTorch.
WARNING: you SHOULD NOT use ".to()" or ".cuda()" in each implementation block.
"""
import torch
from ps1_helper import softmax_loss
from cs639 import Solver


def hello_fully_connected_networks():
    """
    This is a sample function that we will try to import and run to ensure that
    your environment is correctly set up on Google Colab.
    """
    print('Hello from fully_connected_networks.py!')


class Linear(object):

    @staticmethod
    def forward(x, w, b):
        
        """
        Computes the forward pass for an linear (fully-connected) layer.
        The input x has shape (N, d_1, ..., d_k) and contains a minibatch of N
        examples, where each example x[i] has shape (d_1, ..., d_k). We will
        reshape each input into a vector of dimension D = d_1 * ... * d_k, and
        then transform it to an output vector of dimension M.
        Inputs:
        - x: A tensor containing input data, of shape (N, d_1, ..., d_k)
        - w: A tensor of weights, of shape (D, M)
        - b: A tensor of biases, of shape (M,)
        Returns a tuple of:
        - out: output, of shape (N, M)
        - cache: (x, w, b)
        """
        out = None
        ######################################################################
        # TODO: Implement the linear forward pass. Store the result in out.  #
        # You will need to reshape the input into rows.                      #
        ######################################################################
        # Replace "pass" statement with your code
        N = x.shape[0]
        D = w.shape[0]
        # M = b.shape[0]
        x_mid = torch.reshape(x,(N,D))
        out = torch.matmul(x_mid,w) + b
        ######################################################################
        #                        END OF YOUR CODE                            #
        ######################################################################
        cache = (x, w, b)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for an linear layer.
        Inputs:
        - dout: Upstream derivative, of shape (N, M)
        - cache: Tuple of:
          - x: Input data, of shape (N, d_1, ... d_k)
          - w: Weights, of shape (D, M)
          - b: Biases, of shape (M,)
        Returns a tuple of:
        - dx: Gradient with respect to x, of shape
          (N, d1, ..., d_k)
        - dw: Gradient with respect to w, of shape (D, M)
        - db: Gradient with respect to b, of shape (M,)
        """
        x, w, b = cache
        dx, dw, db = None, None, None
        ##################################################
        # TODO: Implement the linear backward pass.      #
        ##################################################
        # Replace "pass" statement with your code
        N = x.shape[0]
        D = w.shape[0]
        M = b.shape[0]
        db = torch.sum(dout,dim=0)
        dx = torch.zeros_like(x)
        dx = torch.reshape(dx,(N,D))
        dx += dout @ w.t()
        dw = torch.zeros_like(w)
        dw += torch.reshape(x,(N,D)).t() @ dout
        dx = torch.reshape(dx,x.shape)
        ##################################################
        #                END OF YOUR CODE                #
        ##################################################
        return dx, dw, db


class ReLU(object):

    @staticmethod
    def forward(x):
        """
        Computes the forward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - x: Input; a tensor of any shape
        Returns a tuple of:
        - out: Output, a tensor of the same shape as x
        - cache: x
        """
        out = None
        ###################################################
        # TODO: Implement the ReLU forward pass.          #
        # You should not change the input tensor with an  #
        # in-place operation.                             #
        ###################################################
        # Replace "pass" statement with your code
        # keep_shape = x.shape
        # x = torch.flatten(x)
        # out = torch.zeros_like(x)
        # for i in range(x.shape[0]):
        #     out[i] = max(x[i],0)
        # out = torch.reshape(out,keep_shape)
        # x = torch.reshape(x,keep_shape)
        out = torch.zeros_like(x)
        out = torch.maximum(x,out)
        ###################################################
        #                 END OF YOUR CODE                #
        ###################################################
        cache = x
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Computes the backward pass for a layer of rectified
        linear units (ReLUs).
        Input:
        - dout: Upstream derivatives, of any shape
        - cache: Input x, of same shape as dout
        Returns:
        - dx: Gradient with respect to x
        """
        dx, x = None, cache
        #####################################################
        # TODO: Implement the ReLU backward pass.           #
        # You should not change the input tensor with an    #
        # in-place operation.                               #
        #####################################################
        # Replace "pass" statement with your code
        dx = torch.zeros_like(x)
        dx = torch.maximum(x,dx)
        dx[dx>0] = 1
        dx = dx * dout
        #####################################################
        #                  END OF YOUR CODE                 #
        #####################################################
        return dx


class Linear_ReLU(object):

    @staticmethod
    def forward(x, w, b):
        """
        Convenience layer that performs an linear transform
        followed by a ReLU.

        Inputs:
        - x: Input to the linear layer
        - w, b: Weights for the linear layer
        Returns a tuple of:
        - out: Output from the ReLU
        - cache: Object to give to the backward pass
        """
        a, fc_cache = Linear.forward(x, w, b)
        out, relu_cache = ReLU.forward(a)
        cache = (fc_cache, relu_cache)
        return out, cache

    @staticmethod
    def backward(dout, cache):
        """
        Backward pass for the linear-relu convenience layer
        """
        fc_cache, relu_cache = cache
        da = ReLU.backward(dout, relu_cache)
        dx, dw, db = Linear.backward(da, fc_cache)
        return dx, dw, db


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.
    The architecure should be linear - relu - linear - softmax.
    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to PyTorch tensors.
    """

    def __init__(self, input_dim=3*32*32, hidden_dim=100, num_classes=10,
                 weight_scale=1e-3, reg=0.0,
                 dtype=torch.float32, device='cpu'):
        """
        Initialize a new network.
        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.params = {}
        self.reg = reg

        ###################################################################
        # TODO: Initialize the weights and biases of the two-layer net.   #
        # Weights should be initialized from a Gaussian centered at       #
        # 0.0 with standard deviation equal to weight_scale, and biases   #
        # should be initialized to zero. All weights and biases should    #
        # be stored in the dictionary self.params, with first layer       #
        # weights and biases using the keys 'W1' and 'b1' and second layer#
        # weights and biases using the keys 'W2' and 'b2'.                #
        ###################################################################
        # Replace "pass" statement with your code
        self.params['W1'] = weight_scale * torch.randn(size=(input_dim,hidden_dim),dtype=dtype,device=device)
        # self.params['W1'] = torch.normal(mean=0.0, std=weight_scale,size=(input_dim,hidden_dim),dtype=dtype).to(device)
        self.params['b1'] = torch.zeros(hidden_dim,dtype=dtype).to(device)
        self.params['W2'] = weight_scale * torch.randn(size=(hidden_dim,num_classes),dtype=dtype,device=device)
        #self.params['W2'] = torch.normal(mean=0.0, std=weight_scale,size=(hidden_dim,num_classes),dtype=dtype).to(device)
        self.params['b2'] = torch.zeros(num_classes,dtype=dtype).to(device)
        ###############################################################
        #                            END OF YOUR CODE                 #
        ###############################################################

    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'params': self.params,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.reg = checkpoint['reg']
        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)
        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Tensor of input data of shape (N, d_1, ..., d_k)
        - y: int64 Tensor of labels, of shape (N,). y[i] gives the
          label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model
        and return:
        - scores: Tensor of shape (N, C) giving classification scores,
          where scores[i, c] is the classification score for X[i]
          and class c.
        If y is not None, then run a training-time forward and backward
        pass and return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping
          parameter names to gradients of the loss with respect to
          those parameters.
        """
        scores = None
        #############################################################
        # TODO: Implement the forward pass for the two-layer net,   #
        # computing the class scores for X and storing them in the  #
        # scores variable.                                          #
        #############################################################
        # Replace "pass" statement with your code
        layer1 = Linear_ReLU()
        layer2 = Linear()
        out1, cache1 = layer1.forward(X,self.params['W1'],self.params['b1'])
        out2, cache2 = layer2.forward(out1,self.params['W2'],self.params['b2'])
        #print("Out2:",out2)
        scores = out2.detach()
        ##############################################################
        #                     END OF YOUR CODE                       #
        ##############################################################

        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ###################################################################
        # TODO: Implement the backward pass for the two-layer net.        #
        # Store the loss in the loss variable and gradients in the grads  #
        # dictionary. Compute data loss using softmax, and make sure that #
        # grads[k] holds the gradients for self.params[k]. Don't forget   #
        # to add L2 regularization!                                       #
        #                                                                 #
        # NOTE: To ensure that your implementation matches ours and       #
        # you pass the automated tests, make sure that your L2            #
        # regularization does not include a factor of 0.5.                #
        ###################################################################
        # Replace "pass" statement with your code
        loss_wo_L2, grad = softmax_loss(out2,y)
        loss = loss_wo_L2 + self.reg * (torch.sum(torch.square(self.params['W1']))
                                        +torch.sum(torch.square(self.params['W2'])))
        dx2,dw2,db2 =layer2.backward(grad,cache2)
        
        dx1,dw1,db1 =layer1.backward(dx2,cache1)
        grads['W2'] = dw2 + 2* self.reg * self.params['W2']
        grads['W1'] = dw1 + 2* self.reg * self.params['W1']
        grads['b1'] = db1
        grads['b2'] = db2
        ###################################################################
        #                     END OF YOUR CODE                            #
        ###################################################################

        return loss, grads


class FullyConnectedNet(object):
    """
    A fully-connected neural network with an arbitrary number of hidden layers,
    ReLU nonlinearities, and a softmax loss function.
    For a network with L layers, the architecture will be:

    {linear - relu} x (L - 1) - linear - softmax

    where the {...} block is repeated L - 1 times.

    Similar to the TwoLayerNet above, learnable parameters are stored in the
    self.params dictionary and will be learned using the Solver class.
    """

    def __init__(self, hidden_dims, input_dim=3*32*32, num_classes=10,
                 reg=0.0, weight_scale=1e-2, seed=None,
                 dtype=torch.float, device='cpu'):
        """
        Initialize a new FullyConnectedNet.

        Inputs:
        - hidden_dims: A list of integers giving the size of each
          hidden layer.
        - input_dim: An integer giving the size of the input.
        - num_classes: An integer giving the number of classes to classify.
        - reg: Scalar giving L2 regularization strength.
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - dtype: A torch data type object; all computations will be
          performed using this datatype. float is faster but less accurate,
          so you should use double for numeric gradient checking.
        - device: device to use for computation. 'cpu' or 'cuda'
        """
        self.reg = reg
        self.num_layers = 1 + len(hidden_dims)
        self.dtype = dtype
        self.params = {}

        #######################################################################
        # TODO: Initialize the parameters of the network, storing all         #
        # values in the self.params dictionary. Store weights and biases      #
        # for the first layer in W1 and b1; for the second layer use W2 and   #
        # b2, etc. Weights should be initialized from a normal distribution   #
        # centered at 0 with standard deviation equal to weight_scale. Biases #
        # should be initialized to zero.                                      #
        #######################################################################
        # Replace "pass" statement with your code
        weight_name_list = ['W'+ str(i) for i in range(1,self.num_layers+1)]
        bias_name_list = ['b'+ str(i) for i in range(1,self.num_layers+1)]

        # initialize the first layer
        self.params[weight_name_list[0]] = weight_scale * torch.randn(size=(input_dim,hidden_dims[0]),
                                                                      dtype=dtype,device=device)
        self.params[bias_name_list[0]] = torch.zeros(hidden_dims[0],dtype=dtype,device=device)

        # initialize hidden layers
        for i in range(1,self.num_layers-1):
          self.params[weight_name_list[i]] = weight_scale * torch.randn(size=(hidden_dims[i-1],hidden_dims[i]),
                                                                      dtype=dtype,device=device)
          self.params[bias_name_list[i]] = torch.zeros(hidden_dims[i],dtype=dtype,device=device)

        # initialize the last linear layer
        self.params[weight_name_list[self.num_layers-1]] = weight_scale * torch.randn(size=(hidden_dims[-1],num_classes),
                                                                      dtype=dtype,device=device)
        self.params[bias_name_list[self.num_layers-1]] = torch.zeros(num_classes,dtype=dtype,device=device)
        #######################################################################
        #                         END OF YOUR CODE                            #
        #######################################################################


    def save(self, path):
        checkpoint = {
          'reg': self.reg,
          'dtype': self.dtype,
          'params': self.params,
          'num_layers': self.num_layers,
        }

        torch.save(checkpoint, path)
        print("Saved in {}".format(path))

    def load(self, path, dtype, device):
        checkpoint = torch.load(path, map_location='cpu')
        self.params = checkpoint['params']
        self.dtype = dtype
        self.reg = checkpoint['reg']
        self.num_layers = checkpoint['num_layers']

        for p in self.params:
            self.params[p] = self.params[p].type(dtype).to(device)

        print("load checkpoint file: {}".format(path))

    def loss(self, X, y=None):
        """
        Compute loss and gradient for the fully-connected net.
        Input / output: Same as TwoLayerNet above.
        """
        X = X.to(self.dtype)
        mode = 'test' if y is None else 'train'

        scores = None
        ##################################################################
        # TODO: Implement the forward pass for the fully-connected net,  #
        # computing the class scores for X and storing them in the       #
        # scores variable.                                               #
        #                                                                #
        #                                                                #
        #                                                                #
        ##################################################################
        # Replace "pass" statement with your code
        w_name_list = ['W'+ str(i) for i in range(1,self.num_layers+1)]
        b_name_list = ['b'+ str(i) for i in range(1,self.num_layers+1)]

        layer_list = [Linear_ReLU() for i in range(self.num_layers-1)]
        layer_list.append(Linear()) 
        temp_out = None
        cache_list = []
        temp_cache = None

        # start the forward pass
        temp_out, temp_cache = layer_list[0].forward(X,self.params['W1'],self.params['b1'])
        cache_list.append(temp_cache)
        for i in range(1,self.num_layers):
          temp_out, temp_cache = layer_list[i].forward(temp_out,self.params[w_name_list[i]],
                                                       self.params[b_name_list[i]])
          cache_list.append(temp_cache)
        #res_out, res_cache = layer2.forward(out1,self.params['W2'],self.params['b2'])
        #print("Out2:",out2)
        scores = temp_out.detach()
        #################################################################
        #                      END OF YOUR CODE                         #
        #################################################################

        # If test mode return early
        if mode == 'test':
            return scores

        loss, grads = 0.0, {}
        #####################################################################
        # TODO: Implement the backward pass for the fully-connected net.    #
        # Store the loss in the loss variable and gradients in the grads    #
        # dictionary. Compute data loss using softmax, and make sure that   #
        # grads[k] holds the gradients for self.params[k]. Don't forget to  #
        # add L2 regularization!                                            #
        # NOTE: To ensure that your implementation matches ours and you     #
        # pass the automated tests, make sure that your L2 regularization   #
        # includes a factor of 0.5 to simplify the expression for           #
        # the gradient.                                                     #
        #####################################################################
        # Replace "pass" statement with your code
        loss_wo_L2, grad = softmax_loss(temp_out,y)
        reg_loss = 0
        for i in range(self.num_layers):
            reg_loss += torch.sum(torch.square(self.params[w_name_list[i]]))
        loss = loss_wo_L2 + 0.5 * self.reg * reg_loss

        # start back propagation
        #temp_dw, temp_dx, temp_db = None
        temp_dx, temp_dw, temp_db =layer_list[-1].backward(grad,cache_list[-1])
        grads[b_name_list[-1]] = temp_db
        grads[w_name_list[-1]] = temp_dw + self.reg * self.params[w_name_list[-1]]
        for i in range(self.num_layers-2,-1,-1):
            temp_dx, temp_dw, temp_db =layer_list[i].backward(temp_dx,cache_list[i])
            grads[b_name_list[i]] = temp_db
            grads[w_name_list[i]] = temp_dw + self.reg * self.params[w_name_list[i]]

        # dx1,dw1,db1 =layer1.backward(dx2,cache1)
        # grads['W2'] = dw2 + 2* self.reg * self.params['W2']
        # grads['W1'] = dw1 + 2* self.reg * self.params['W1']
        # grads['b1'] = db1
        # grads['b2'] = db2
        ###########################################################
        #                   END OF YOUR CODE                      #
        ###########################################################

        return loss, grads


def create_solver_instance(data_dict, dtype, device):
    model = TwoLayerNet(hidden_dim=200, dtype=dtype, weight_scale=1e-2,device=device)
    #############################################################
    # TODO: Use a Solver instance to train a TwoLayerNet that   #
    # achieves at least 50% accuracy on the validation set.     #
    #############################################################
    solver = None
    # Replace "pass" statement with your code
    model.reg = 0.001
    My_data = {
      'X_train': data_dict['X_train'],
      'y_train': data_dict['y_train'],
      'X_val': data_dict['X_val'],
      'y_val': data_dict['y_val']
    }
    optim_configs = {}
    solver = Solver(model,My_data,
                    device='cuda',
                    print_every=1000,
                    num_epochs=120, 
                    batch_size=128,
                    print_acc_every=4,
                    lr_decay = 1,
                    optim_config={
              'learning_rate': 5*1e-3,
            },)
    ##############################################################
    #                    END OF YOUR CODE                        #
    ##############################################################
    return solver


def get_three_layer_network_params():
    ###############################################################
    # TODO: Change weight_scale and learning_rate so your         #
    # model achieves 100% training accuracy within 20 epochs.     #
    ###############################################################
    weight_scale = 0.09   # Experiment with this!
    learning_rate = 0.11  # Experiment with this!
    # Replace "pass" statement with your code
    pass
    ################################################################
    #                             END OF YOUR CODE                 #
    ################################################################
    return weight_scale, learning_rate


def get_five_layer_network_params():
    ################################################################
    # TODO: Change weight_scale and learning_rate so your          #
    # model achieves 100% training accuracy within 20 epochs.      #
    ################################################################
    learning_rate = 0.09 # Experiment with this!
    weight_scale = 0.12   # Experiment with this!
    # Replace "pass" statement with your code
    pass
    ################################################################
    #                       END OF YOUR CODE                       #
    ################################################################
    return weight_scale, learning_rate




