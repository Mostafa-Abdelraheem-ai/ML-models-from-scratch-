class LayerDens:
    # weights = n_inputs * n_neurons
    # inputs = n_sampels * n_inputs
    # output = n_sampels * n_calsses
    # d_values = n_sampels * n_nurons    
    
    def __init__(self, n_inputs, n_neurons):
        self.weights = .01 * np.random.randn(n_inputs, n_neurons)
        self.biases = np.zeros((1, n_neurons))
        
    def forward (self, inputs):
        self.inputs = inputs
        self.outputs = np.dot(inputs, self.weights) + self.biases
        
    def backward (self, dvalues):
        self.dwieghts = np.dot(self.inputs.T, dvalues)
        self.dbais = np.sum(dvalues,axis = 0 ,keepdims=True)
        self.dinputs = np.dot(dvalues, self.weights.T) 
        
 
class ActivationSigmoid:
    def forward (self, inputs):
        self.inputs = inputs
        self.outputs = 1 / (1 + np.exp(-inputs))     
        
    def backword (self, dvalues):
        self.dinputs = self.outputs * (1-self.outputs) * dvalues

class ActivationReLU:
    def forward (self, inputs):
        self.inputs = inputs
        self.outputs = np.maximum(0,inputs)
        
    def backword (self, dvalues):
        self.dinputs = dvalues.copy()
        self.dinputs[self.inputs <= 0 ] = 0 
        

class ActivationSoftMax:
    def forward (self, inputs):
        exp_val = np.exp(inputs-np.max(inputs,axis=1,keepdims=True))
        self.outputs = exp_val / np.sum(exp_val, axis=1, keepdims=True)
        
    def backword (self, dvalues):
        self.dinput = np.empty_like(dvalues)
        for index, (single_input, single_dvalue) in enumerate(zip(self.outputs,dvalues)):
            single_output = single_output.reshape(-1,1)
            jacobian_matrix = np.diagflat(single_output) - np.dot(single_output, single_output.T)
            self.outputs[index] = np.dot(jacobian_matrix, single_dvalue)          
        
        
class Loss :
    def calculate(self, output, y):
        sample_losses = self.forward(output, y)
        data_loss = np.mean(sample_losses)
        return data_loss

class LossCategoricalCrossEntopy(Loss):
    def forward (self, y_pred, y_true):
        sampels = len(y_true)
        y_pred = np.clip(y_pred, 1e-7, 1)
        #for label encoded values
        if len(y_true.shape) == 1:
            correct_prob = y_pred[range(sampels), y_true]
        #for one hot encoded label values
        elif len(y_true.shape) == 2:
            correct_prob = np.sum(y_pred * y_true,axis=1)
            
        cross_entropy = -np.log(correct_prob)
        return cross_entropy    
        
    def backword (self, y_true, y_hat):
        self.dinputs = - y_true / y_hat        

class ActivationSoftMax_LossCategoricalCrossEntopy :
    def __init__(self):
        self.activation = ActivationSoftMax()
        self.loss = LossCategoricalCrossEntopy()
        
        
    def forward(self, inputs, y_true):
        self.activation.forward(inputs)
        self.outputs = self.activation.outputs
        return self.loss.calculate(self.outputs, y_true)
        
    def backword (self, y_pred, y_true):
        sampels = len(y_pred)
        self.dinputs = y_pred.copy()
        
        if len(y_true.shape) == 2:
            y_true = np.argmax(y_true, axis = 1)
            
        self.dinputs[range(sampels), y_true] -= 1
        self.dinputs = self.dinputs / sampels 
            
class Optimizer_SGD:
    def __init__(self, learning_rate = 1.0, decay = 0, momentum = 0):
        self.decay = decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iteration = 0
        self.momentum = momentum
        
    def pre_update(self): 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iteration  ))    
    
    def update_params(self, layer):
        if self. momentum :
            if not hasattr(layer, "weight_momentums"):
                layer.weight_momentums = np.zeros_like(layer.weights)
                layer.bias_momentums = np.zeros_like(layer.biases)
            
            wight_update = -self.current_learning_rate * layer.dwieghts +  self.momentum * layer.weight_momentums
            layer.weight_momentums = wight_update
            
            bias_update = -self.current_learning_rate * layer.dbais + self.momentum * layer.bias_momentums
            layer.bias_momentums = bias_update
            
            layer.weights += wight_update
            layer.biases += bias_update
            
        else :
            layer.weights -= self.current_learning_rate * layer.dwieghts
            layer.biases -= self.current_learning_rate * layer.dbais
            
    def post_update(self): 
        self.iteration += 1
        
class Optimizer_Adagrad:
    def __init__(self, learning_rate = 1.0, decay = 0, epslion = 1e-7):
        self.decay = decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iteration = 0
        self.epslion = epslion
        
    def pre_update(self): 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iteration  ))    
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cashe"):
            layer.weight_cashe = np.zeros_like(layer.weights)
            layer.bias_cashe = np.zeros_like(layer.biases)
        
        layer.weight_cashe += layer.dwieghts ** 2
        layer.bias_cashe += layer.dbais ** 2
        
        layer.weights -= self.current_learning_rate * layer.dwieghts / (np.sqrt((layer.weight_cashe))+self.epslion)
        layer.biases -= self.current_learning_rate * layer.dbais / (np.sqrt((layer.bias_cashe ))+ self.epslion) 

            
    def post_update(self): 
        self.iteration += 1
        
        
class Optimizer_RMSprop:
    def __init__(self, learning_rate = .001, decay = 0, epslion = 1e-7, rho = 0.9):
        self.decay = decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iteration = 0
        self.epslion = epslion
        self.rho = rho
        
    def pre_update(self): 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iteration  ))    
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cashe"):
            layer.weight_cashe = np.zeros_like(layer.weights)
            layer.bias_cashe = np.zeros_like(layer.biases)
        
        layer.weight_cashe = self.rho * layer.weight_cashe + (1-self.rho) * layer.dwieghts ** 2
        layer.bias_cashe = self.rho * layer.bias_cashe + (1-self.rho) * layer.dbais ** 2
        
        layer.weights -= self.current_learning_rate * layer.dwieghts / (np.sqrt(layer.weight_cashe)+self.epslion)
        layer.biases -= self.current_learning_rate * layer.dbais / (np.sqrt(layer.bias_cashe )+ self.epslion) 

            
    def post_update(self): 
        self.iteration += 1
        
        
class Optimizer_Adam:
    def __init__(self, learning_rate = .001, decay = 0, epslion = 1e-7, beta1 = 0.9, beta2 = .999):
        self.decay = decay
        self.learning_rate = learning_rate
        self.current_learning_rate = learning_rate
        self.iteration = 0
        self.epslion = epslion
        self.beta1 = beta1
        self.beta2  =beta2        
    def pre_update(self): 
        if self.decay:
            self.current_learning_rate = self.learning_rate * (1 / (1 + self.decay * self.iteration  ))    
    
    def update_params(self, layer):
        if not hasattr(layer, "weight_cashe"):
            layer.weight_cashe = np.zeros_like(layer.weights)
            layer.bias_cashe = np.zeros_like(layer.biases)
            layer.weight_momentums = np.zeros_like(layer.weights)
            layer.bias_momentums = np.zeros_like(layer.biases)
        
        layer.weight_momentums = self.beta1 * layer.weight_momentums + (1 - self.beta1) * layer.dwieghts
        layer.bias_momentums = self.beta1 * layer.bias_momentums + (1- self.beta1) * layer.dbais
        
        weights_momentums_corrected = layer.weight_momentums / (1 - self.beta1 ** (self.iteration+1))
        bias_momentums_corrected = layer.bias_momentums / (1 - self.beta1 ** (self.iteration+1))
        
        layer.weight_cashe = self.beta2 * layer.weight_cashe + (1-self.beta2) * layer.dwieghts ** 2
        layer.bias_cashe = self.beta2 * layer.bias_cashe + (1-self.beta2) * layer.dbais ** 2
        
        weights_cashe_corrected = layer.weight_cashe / (1 - self.beta2 ** (self.iteration+1))
        bias_cashe_corrected = layer.bias_cashe / (1 - self.beta2 ** (self.iteration+1))
        
        layer.weights -= self.current_learning_rate * weights_momentums_corrected / (np.sqrt(weights_cashe_corrected) + self.epslion)
        layer.biases -= self.current_learning_rate * bias_momentums_corrected / (np.sqrt(bias_cashe_corrected) + self.epslion) 

            
    def post_update(self): 
        self.iteration += 1
        
    