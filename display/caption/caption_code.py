
import tensorflow as tf
import matplotlib.pyplot as plt
import collections
import random
import numpy as np
import os
import time
import json
from PIL import Image
# annotation_folder = '/annotations/'
# if not os.path.exists(os.path.abspath('.') + annotation_folder):
#   annotation_zip = tf.keras.utils.get_file('captions.zip',
#                                            cache_subdir=os.path.abspath('.'),
#                                            origin='http://images.cocodataset.org/annotations/annotations_trainval2014.zip',
#                                            extract=True)
#   annotation_file = os.path.dirname(annotation_zip)+'/annotations/captions_train2014.json'
#   os.remove(annotation_zip)
# image_folder = '/train2014/'
# if not os.path.exists(os.path.abspath('.') + image_folder):
#   image_zip = tf.keras.utils.get_file('train2014.zip',
#                                       cache_subdir=os.path.abspath('.'),
#                                       origin='http://images.cocodataset.org/zips/train2014.zip',
#                                       extract=True)
#   PATH = os.path.dirname(image_zip) + image_folder
#   os.remove(image_zip)
# else:
#   PATH = os.path.abspath('.') + image_folder
# image_path_to_caption = collections.defaultdict(list)
# from pathlib import Path
# annotation_file =Path('annotations/captions_train2014.json')
# with open(annotation_file, 'r') as f:
#     annotations = json.load(f)
# for val in annotations['annotations']:
#   caption = f"<start> {val['caption']} <end>"
#   image_path = PATH + '/COCO_train2014_' + '%012d.jpg' % (val['image_id'])
#   image_path_to_caption[image_path].append(caption)


train_captions = []
# img_name_vector = []


# for image_path in image_paths:
#   caption_list = image_path_to_caption[image_path]
#   train_captions.extend(caption_list)
#   img_name_vector.extend([image_path] * len(caption_list))
# BATCH_SIZE = 64
# BUFFER_SIZE = 1000
caption_dataset = tf.data.Dataset.from_tensor_slices(train_captions)
embedding_dim = 256
units = 512
# Max word count for a caption.
max_length = 50
def standardize(inputs):
  inputs = tf.strings.lower(inputs)
  return tf.strings.regex_replace(inputs,
                                  r"!\"#$%&\(\)\*\+.,-/:;=?@\[\\\]^_`{|}~", "")
vocabulary_size = None
tokenizer = tf.keras.layers.TextVectorization(
    max_tokens=vocabulary_size,
    standardize=standardize,
    output_sequence_length=max_length)
# Learn the vocabulary from the caption data.
tokenizer.adapt(caption_dataset)
# num_steps = len(img_name_train) // BATCH_SIZE
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

class CNN_Encode(tf.keras.Model):
    # Since you have already extracted the features and dumped it
    # This encoder passes those features through a Fully connected layer
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        # shape after fc == (batch_size, 64, embedding_dim)
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x
class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    # features(CNN_encoder output) shape == (batch_size, 64, embedding_dim)

    # hidden shape == (batch_size, hidden_size)
    # hidden_with_time_axis shape == (batch_size, 1, hidden_size)
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

class RNN_Decode(tf.keras.Model):
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
    def __call__(self, shape, dtype=None):
        return custom_initializer(shape, dtype=dtype)
from keras.utils.generic_utils import get_custom_objects
attention_features_shape = 64
max_length = 50
# encoder = CNN_Encoder(embedding_dim)

# decoder = RNN_Decoder(embedding_dim, units, tokenizer.vocabulary_size())
encoder = tf.keras.models.load_model("C:/Users/jdalm/Desktop/amalitech projects/image captioning/display/caption/models/encoder", custom_objects  = {"CustomModel": CNN_Encode})

decoder = tf.keras.models.load_model("C:/Users/jdalm/Desktop/amalitech projects/image captioning/display/caption/models/decoder", custom_objects = {"reset_state": RNN_Decode })
get_custom_objects().update({'custom_objects': RNN_Decode.reset_state})
def evaluate(image):
    attention_plot = np.zeros((max_length, attention_features_shape))

    hidden = decoder.reset_state(batch_size=1)

    temp_input = tf.expand_dims(load_image(image)[0], 0)
    img_tensor_val = model(temp_input)
    img_tensor_val = tf.reshape(img_tensor_val, (img_tensor_val.shape[0],
                                                 -1,
                                                 img_tensor_val.shape[3]))

    features = encoder(img_tensor_val)

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
  temp_image = np.array(Image.open(image))



















































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
    path = 'image/'
# image_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/e/e9/Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png/300px-Felis_silvestris_silvestris_small_gradual_decrease_of_quality.png'
    # path = path + image_url
    # image_extension = image_url[-4:]
    # image_path = tf.keras.utils.get_file('image'+image_extension,origin=path)
    img = Image.open(image_url)
    result, attention_plot = evaluate(img)
    print('Prediction Caption:', ' '.join(result))
    plot_attention(img, result, attention_plot)
    # opening the image
    Image.open(img)