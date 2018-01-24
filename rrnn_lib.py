import numpy as np
import tensorflow as tf
from jive_data_utils import DataUtils
from flask import Flask, jsonify, render_template, request


class RetentionPredictor:

    def __init__(self, save_dir = 'save/save_point_final_2'):
        print("Creating tf session")
        self.sess = tf.Session()
        self.saver = tf.train.import_meta_graph(save_dir + '.meta')
        self.saver.restore(self.sess, save_dir)
        self.graph = tf.get_default_graph()
        self.save_dir = 'save/save_point_3'
        self.learning_rate_val = 0.001
        print("Trying to restore session")
        self.data_utils = DataUtils()
        
        
    def predict(self,x):
        x = self.data_utils.scale_data(x)
        batch = [[x]]
        input_data, initial_state, final_state, targets, keep_prob, accurracy, predictions, learning_rate, is_training = self.get_tensors()

        prev_state = self.sess.run(initial_state)
        feed = {
                    input_data: batch,
                    keep_prob: 1,
                    initial_state: prev_state,
                    learning_rate: self.learning_rate_val,
                    is_training: False
               }
        
        pred = self.sess.run([predictions], feed_dict=feed)

        return np.asarray(pred).flatten()

    def get_tensors(self):
        """
        Get input, initial state, final state, keep_prob, accurracy and predictions tensor from <loaded_graph>
        :param loaded_graph: TensorFlow graph loaded from file
        :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, KeepProbTensor, AccurracyTensor, PredictionsTensor)
        """
        # TODO: Implement Function
        inputs = self.graph.get_tensor_by_name("inputs:0")
        targets = self.graph.get_tensor_by_name("targets:0")
        initial_state = self.graph.get_tensor_by_name("initial_state:0")
        final_state = self.graph.get_tensor_by_name("final_state:0")
        keep_prob = self.graph.get_tensor_by_name("keep_prob:0")
        accurracy = self.graph.get_tensor_by_name("accurracy:0")
        predictions = self.graph.get_tensor_by_name("predictions:0")
        learning_rate = self.graph.get_tensor_by_name("learning_rate:0")
        is_training = self.graph.get_tensor_by_name("is_training:0")
        return inputs, initial_state, final_state, targets, keep_prob, accurracy, predictions, learning_rate, is_training


'''
sess, graph = restore_session()



output = prediction(data,sess, graph)

print(output)
'''

#data = [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,-0.12161573,-0.44953505,0.459073]
data = [0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0.0049808333,2.5,9]
#data = [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0.0049785, 2.5, 3]
data_utils = DataUtils()

#d = data_utils.scale_data(data)
#print(d)
#print(d)
predictor = RetentionPredictor()
output = predictor.predict(data)

print(output)