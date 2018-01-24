
# coding: utf-8

# ## Retention Team - Hackathon project

# In[1]:

get_ipython().magic('matplotlib inline')
get_ipython().magic("config InlineBackend.figure_format = 'retina'")

import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# #### Load and prepare data

# In[2]:

data_path = 'data/customers.csv'
data_path_final = 'data/customers-final-2.csv'

customers = pd.read_csv(data_path)
customers_final = pd.read_csv(data_path_final)


fields_to_drop = ['Unnamed: 0','MRR', 'Seats','days']
data = customers_final.drop(fields_to_drop, axis=1)


# In[3]:

data.head()


# #### Checking data

# In[4]:

data.shape


# ### Scatter matrix

# ### Preparing Data
# 

# In[5]:

data.head()


# #### Scaling variables
# 

# In[6]:

quant_features = ['time_delta','Tickets','csat']

scaled_features = {}
for each in quant_features:
    mean, std = data[each].mean(), data[each].std()
    scaled_features[each] = [mean, std]
    data.loc[:, each] = (data[each] - mean)/std


# In[7]:

data.head()


# ### Splitting the data into training, testing and validation sets

# In[8]:

def split_data(data, val_size=0.2, test_size=0.2):
    ntest = int(round(len(data)* (1 - test_size)))
    nval = int(round(len(data.iloc[:ntest])* (1-val_size)))
    
    df_train, df_val, df_test = data.iloc[:nval], data.iloc[nval:ntest],data.iloc[ntest:]
    
    return df_train, df_val, df_test


# In[9]:

train_data, val_data, test_data = split_data(data,0.1,0.1)

target_fields = ['status']


train_features, train_targets = train_data.drop(target_fields, axis=1), train_data[target_fields]

test_features, test_targets = test_data.drop(target_fields, axis=1), test_data[target_fields]
test_features, test_targets = test_features[-21*24:], test_targets[-21*24:]
val_features, val_targets = val_data.drop(target_fields, axis=1), val_data[target_fields]


# In[10]:

train_targets.head()


# In[11]:

print(train_features.shape)
print(val_features.shape)
print(test_features.shape)


# ### Build batches

# In[12]:

# each element of x is an array with 53 features and each element of y is an array with 3 targets
# each x is one hour features 
def get_batches(x, y, n_seqs, n_steps):
    '''Create a generator that returns batches of size
       n_seqs x n_steps from arr.
       
       Arguments
       ---------
       array x and array y: Array you want to make batches from
       n_seqs: Batch size, the number of sequences per batch
       n_steps: Number of sequence steps per batch
    '''
    # Get the number of hours per batch and number of batches we can make
    hours_per_batch = n_seqs * n_steps
    n_batches = len(x)//hours_per_batch
    
    # convert from Pandas to np remove the index column
    x = x.reset_index().values[:,1:]
    y = y.reset_index().values[:,1:]

    # make only full batches    
    x, y = x[:n_batches*hours_per_batch], y[:n_batches*hours_per_batch]

    # TODO: this needs to be optmized
    # x_temp will be ( n rows x n_steps wide) where each element is an array of 53 features
    # this first look splits the x with n rows and n_steps wide 
    x_temp = []
    y_temp = []
    for st in range(0, n_batches*hours_per_batch, n_steps ):
        x_temp.append( x[st:st+n_steps]  )
        y_temp.append( y[st:st+n_steps]  )

    x = np.asarray(x_temp )    
    y = np.asarray(y_temp )    

    # this splits x in n_seqs rows so the return is a batch of n_seqs rows with n_steps wide 
    # where each element is an array of 53 features (one hour from our data)
    for sq in range(0,(n_batches*hours_per_batch)//n_steps, n_seqs ):
        yield x[sq:sq+n_seqs,:,:], y[sq:sq+n_seqs,:,:]


# ### Test batches

# In[13]:

print(train_features.shape)
batches = get_batches(train_features, train_targets,1,5)
x, y = next(batches)
print(x.shape)
print(y.shape)
print(x)
print(y)


# # Building the network

# ## Get inputs

# In[14]:

import tensorflow as tf


# In[15]:

def get_inputs(batch_size, num_features, num_targets):

    # Declare placeholders we'll feed into the graph
    input_data = tf.placeholder(tf.float32, [None,None, num_features], name='inputs')
    targets = tf.placeholder(tf.float32, [None,None, num_targets], name='targets')

    # Keep probability placeholder for drop out layers
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    learningRate = tf.placeholder(tf.float32, name='learning_rate')
    
     # Add placeholder to indicate whether or not we're training the model
    is_training = tf.placeholder(tf.bool, name='is_training')
    
    return input_data,targets,keep_prob,learningRate, is_training


# ### Build LSTM cells

# In[16]:

def lstm_cell(lstm_size, keep_prob):
    cell = tf.contrib.rnn.LSTMCell(lstm_size, initializer=tf.random_uniform_initializer(-0.1, 0.1, seed=2))
    return tf.contrib.rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)

def get_init_cell(batch_size, lstm_size, keep_prob, num_layers):
    cell = tf.contrib.rnn.MultiRNNCell([lstm_cell(lstm_size,keep_prob) for _ in range(num_layers)])
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    return cell, initial_state


# Build RNN with batch normalization

# In[17]:

def build_rnn(cell, input_data, lstm_size, is_training):
    
    outputs, final_state = tf.nn.dynamic_rnn(cell, input_data,  dtype=tf.float32)
    outputs = tf.layers.batch_normalization(outputs, training=is_training)
    final_state = tf.identity(final_state, name='final_state')
    return outputs, final_state


# Build Fully connected layer with batch normalization

# In[18]:

def fully_connected(prev_layer, num_units, is_training):
    """
    Create a fully connectd layer with the given layer as input and the given number of neurons.
    
    :param prev_layer: Tensor
        The Tensor that acts as input into this layer
    :param num_units: int
        The size of the layer. That is, the number of units, nodes, or neurons.
    :param is_training: bool or Tensor
        Indicates whether or not the network is currently training, which tells the batch normalization
        layer whether or not it should update or use its population statistics.
    :returns Tensor
        A new fully connected layer
    """
    layer = tf.layers.dense(prev_layer, num_units, use_bias=True, activation=None)
    layer = tf.layers.batch_normalization(layer, training=is_training)
    layer = tf.nn.relu(layer)
    return layer


# #### Build NN

# We are going to build a RNN with 3 fuilly connected layers as hidden layers with RELU activation function and Batch normalization

# In[19]:

def build_nn(cell, lstm_size, input_data, is_training):
    
    #Build RNN with LSTM cells
    outputs, final_state = build_rnn(cell, input_data, lstm_size, is_training)
    
    hidden_1 = fully_connected(outputs,256, is_training)
    
    hidden_2 = fully_connected(hidden_1,128, is_training )
    
    hidden_3 = fully_connected(hidden_1,64, is_training )

    #weights and biases
    weights = tf.truncated_normal_initializer(mean=0.0, stddev=0.1)
    biases = tf.zeros_initializer()
    #Output layer
    predictions = tf.contrib.layers.fully_connected(hidden_3, 
                                               1, 
                                               activation_fn = tf.sigmoid,
                                               weights_initializer=weights,
                                               biases_initializer=biases)
    
    predictions = tf.identity(predictions, name='predictions')
    
    return predictions, final_state


# Hyper parameters

# In[31]:

num_features = 32
num_targets = 1

epochs = 20
batch_size = 10
# one step for each hour that we want the sequence to remember
num_steps = 64
lstm_size = 512
num_layers = 2
learning_rate_val = 0.0001
keep_prob_val = 0.5
#save_dir = 'save/save_point'
save_dir = 'save/save_point_final_3'


# ## Build the graph

# In[21]:

train_graph = tf.Graph()

with train_graph.as_default():
    
    inputs,targets,keep_prob,learning_rate, is_training = get_inputs(batch_size,num_features,num_targets)
    
    cell, initial_state = get_init_cell(batch_size,lstm_size,keep_prob,num_layers)
    
    predictions, final_state = build_nn(cell, lstm_size, inputs, is_training)
    
    #Loss function
    cost = tf.losses.mean_squared_error(targets, predictions)

    #Optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate).minimize(cost)
    
    correct_pred = tf.equal(tf.cast(tf.round(predictions), tf.int32), tf.cast(tf.round(targets), tf.int32))
    accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))
    
    accuracy = tf.identity(accuracy, name='accurracy')


# ### Training

# In[22]:

val_accuracy=[]
training_loss=[]

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())
    
    iteration = 1
    for e in range(epochs):
        
        state = sess.run(initial_state)
        batches = get_batches(train_features, train_targets, batch_size,num_steps)
        for ii, (x, y) in enumerate(batches, 1):

            feed = {inputs: x,
                    targets: y,
                    keep_prob: keep_prob_val,
                    initial_state: state,
                    learning_rate: learning_rate_val,
                    is_training: True}
            
            loss, state, _ = sess.run([cost, final_state, optimizer], feed_dict=feed)

            if iteration%5==0:
                training_loss.append(loss)

            
            if iteration%5==0:
                val_acc = []
                val_state = sess.run(cell.zero_state(batch_size, tf.float32))
                for x, y in get_batches(val_features, val_targets, batch_size,num_steps):
                    feed = {inputs: x,
                            targets: y,
                            keep_prob: 1,
                            initial_state: val_state,
                            learning_rate: learning_rate_val,
                            is_training: False}
                    batch_acc, val_state = sess.run([accuracy, final_state], feed_dict=feed)
                    val_acc.append(batch_acc)
                val_accuracy.append( np.mean(val_acc) )
                

                sys.stdout.write("\rProgress: {:2.1f}".format(100 * e/float(epochs))                         + "% ... Iterations: " + str(iteration)                         + " ... Training loss: " + str(loss)[:]                         + " ... Validation accurracy: " + str(np.mean(val_acc) )[:6])
                sys.stdout.flush()
    
            iteration +=1
    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)


# In[23]:

plt.plot(val_accuracy, label='Accuracy')
plt.legend()
_ = plt.ylim()


# In[24]:

plt.plot(training_loss, label='Loss')
plt.legend()
_ = plt.ylim()


# ## Check out your predictions
# 
# Here, use the test data to view how well your network is modeling the data. If something is completely wrong here, make sure each step in your network is implemented correctly.

# In[25]:

def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, keep_prob, accurracy and predictions tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, KeepProbTensor, AccurracyTensor, PredictionsTensor)
    """
    # TODO: Implement Function
    inputs = loaded_graph.get_tensor_by_name("inputs:0")
    targets = loaded_graph.get_tensor_by_name("targets:0")
    initial_state = loaded_graph.get_tensor_by_name("initial_state:0")
    final_state = loaded_graph.get_tensor_by_name("final_state:0")
    keep_prob = loaded_graph.get_tensor_by_name("keep_prob:0")
    accurracy = loaded_graph.get_tensor_by_name("accurracy:0")
    predictions = loaded_graph.get_tensor_by_name("predictions:0")
    learning_rate = loaded_graph.get_tensor_by_name("learning_rate:0")
    is_training = loaded_graph.get_tensor_by_name("is_training:0")
    return inputs, initial_state, final_state, targets, keep_prob, accurracy, predictions, learning_rate, is_training


# ### Create a graph to compare the data and predictions

# In[27]:

test_acc = []
loaded_graph = tf.Graph()
save_dir = 'save/save_point_final_2'
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(save_dir + '.meta')
    loader.restore(sess, save_dir)
    
    # Get Tensors from loaded model
    input_data, initial_state, final_state, targets, keep_prob, accurracy, predictions, learning_rate, is_training = get_tensors(loaded_graph)
    val_acc = []
    val_state = sess.run(cell.zero_state(batch_size, tf.float32))
    for x, y in get_batches(test_features, test_targets, batch_size,50):
        feed = {input_data: x,
                targets: y,
                keep_prob: 1,
                initial_state: val_state,
                learning_rate: learning_rate_val,
                is_training: False}
        batch_acc, val_state = sess.run([accurracy, final_state], feed_dict=feed)
        val_acc.append(batch_acc)
        

    print("Test accuracy: {:.3f}".format(np.mean(val_acc)))
    
    batch = get_batches(test_features, test_targets, 1, 10)
    x,y = next(batch)
    feed = {input_data: x,
                targets: y,
                keep_prob: 1,
                initial_state: val_state,
                learning_rate: learning_rate_val,
                is_training: False
           }
    
    pred = sess.run([predictions], feed_dict=feed)


# In[28]:

pred = np.asarray(pred)


# In[29]:

pred_flat = pred.flatten()


# In[30]:

print(pred_flat)


# In[120]:

print(test_targets[:10])


# In[ ]:



