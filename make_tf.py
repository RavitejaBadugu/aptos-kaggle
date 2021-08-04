from tensorflow.keras.models import load_model
import tensorflow as tf
import math
from tensorflow.keras.layers import MaxPooling2D

class SPP(tf.keras.layers.Layer):
    def __init__(self,pool_list=[1,2,4],**kwargs):
        super(SPP,self).__init__()
        self.pool_list=pool_list
        self.flatten=tf.keras.layers.Flatten()
        self.concat=tf.keras.layers.Concatenate()
    def get_config(self,):
        config = super().get_config().copy()
        config.update({'self.pool_list':[1,2,4],
                      'self.flatten':tf.keras.layers.Flatten(),
                      'self.concat':tf.keras.layers.Concatenate()})
        return config
        
    def call(self,inputs):
        shape=inputs.get_shape().as_list()
        self.output_list=[]
        for i in self.pool_list:
            pool=int(math.ceil(shape[1]/i))
            stride=int(math.floor(shape[1]/i))
            x=MaxPooling2D((pool,pool),strides=(stride,stride))(inputs)
            x=self.flatten(x)
            self.output_list.append(x)
        output=self.concat(self.output_list)
        return output


inf_model=load_model('dense_spp.h5',custom_objects={'SPP':SPP})
tf.saved_model.save(inf_model,'serving/models/1/')
print('model saved')