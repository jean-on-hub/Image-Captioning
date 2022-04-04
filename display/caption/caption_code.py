
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
import pickle


image_model = tf.keras.applications.InceptionV3(include_top=False,
                                                weights='imagenet')
new_input = image_model.input
hidden_layer = image_model.layers[-1].output

model = tf.keras.Model(new_input, hidden_layer)

train_captions = []

embedding_dim = 256
units = 512

max_length = 50

def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
vocabulary_size = 10000
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)
vect = tf.keras.models.Sequential()
vect.add(tf.keras.Input(shape=(1,), dtype=tf.string))
vect.add(tokenizer)
weights = np.load('caption/vectorizer2/vectorizer.npy',allow_pickle=True)    

vect.layers[0].set_weights(weights)

tokenizer = vect.layers[0]

word_to_index = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary())
index_to_word = tf.keras.layers.StringLookup(
    mask_token="",
    vocabulary=tokenizer.get_vocabulary(),
    invert=True)
features_shape = 2048
attention_features_shape = 64
optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask


  return tf.reduce_mean(loss_)

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)
    @tf.function
    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
    def get_config(self):
      return {"fc": self.fc}

    @classmethod
    def from_config(cls, config):
        return cls(**config)
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)
  

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    # attention_hidden_layer shape == (batch_size, 64, units)
    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    # score shape == (batch_size, 64, 1)
    # This gives you an unnormalized score for each image feature.
    score = self.V(attention_hidden_layer)

    # attention_weights shape == (batch_size, 64, 1)
    attention_weights = tf.nn.softmax(score, axis=1)

    # context_vector shape after sum == (batch_size, hidden_size)
    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights
  def get_config(self):
      return {"W1": self.W1,"W2": self.W2, "V": self.V}
  @classmethod
  def from_config(cls, config):
    return cls(**config)

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)
  @tf.function
  def call(self, x, features, hidden):
    # defining attention as a separate model
    context_vector, attention_weights = self.attention(features, hidden)

    # x shape after passing through embedding == (batch_size, 1, embedding_dim)
    x = self.embedding(x)

    # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

    # passing the concatenated vector to the GRU
    output, state = self.gru(x)

    # shape == (batch_size, max_length, hidden_size)
    x = self.fc1(output)

    # x shape == (batch_size * max_length, hidden_size)
    x = tf.reshape(x, (-1, x.shape[2]))

    # output shape == (batch_size * max_length, vocab)
    x = self.fc2(x)

    return x, state, attention_weights
  
  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))
  
  def get_config(self):
      return {"units": self.units, "embedding": self.embedding,"gru": self.gru, 
              "fc1": self.fc1, "fc2": self.fc2, "attention":self.attention}
  @classmethod
  def from_config(cls, config):
        return cls(**config)
# from keras.utils.generic_utils import get_custom_objects
attention_features_shape = 64
max_length = 50

def load_image(image_path):
    # img = tf.io.read_file(image_path)
    # img = tf.io.decode_jpeg(image_path, channels=3)
    img = tf.keras.layers.Resizing(299, 299)(image_path)
    img = tf.keras.applications.inception_v3.preprocess_input(img)
    return img, image_path
encoder = CNN_Encoder(embedding_dim)

decoder = RNN_Decoder(embedding_dim, units, vocabulary_size)
encoder.load_weights('caption/weigths2/encoder')

decoder.load_weights('caption/weigths2/decoder')
# get_custom_objects().update({'custom_objects': RNN_Decode.reset_state})
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))
    # decoder = RNN_Decoder(embedding_dim, units, vocabulary_size)
    
    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)
    # decoder.load_weights('caption/weigths2/decoder')
    dec_input = tf.expand_dims([word_to_index('<start>')], 0)
    result = []
    
    for i in range(max_length):
        predictions, hidden, attention_weights = decoder(dec_input,
                                                         features,
                                                         hidden)
        

        attention_plot[i] = tf.reshape(attention_weights, (-1, )).numpy()

        predicted_id = tf.random.categorical(predictions, 1)[0][0].numpy()
        predicted_word = tf.compat.as_text(index_to_word(predicted_id).numpy())
        result.append(predicted_word)

        if predicted_word == '<end>':
            return result, attention_plot

        dec_input = tf.expand_dims([predicted_id], 0)

    attention_plot = attention_plot[:len(result), :]
    return result, attention_plot


def plot_attention(image, result, attention_plot):
  temp_image = np.array(image)


  fig = plt.figure(figsize=(30, 30))

  len_result = len(result)
  for i in range(len_result):
      temp_att = np.resize(attention_plot[i], (8, 8))
      grid_size = max(int(np.ceil(len_result/2)), 2)
      ax = fig.add_subplot(grid_size, grid_size, i+1)
      ax.set_title(result[i])
      img = ax.imshow(temp_image)
      ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

  plt.tight_layout()
  plt.show()
def detect(image_url):
  ans = {}
  path = 'image/'
  img = Image.open(image_url)
  result, attention_plot = evaluate(img)
  return(result)
