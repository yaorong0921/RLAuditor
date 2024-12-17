# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""This is the main file to create the toy dataset and clusters."""
#  lint as: python3
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
# from toy.toy_helper import toy_helper



def create_dataset(skip, n_sample=60000):
  """Creates toy dataset and save to disk."""
  assert n_sample > 15 # or it will clearly bug
  concept = np.reshape(np.random.randint(2, size=15 * n_sample),
                       (-1, 15)).astype(np.bool_) # shape (n_sample, 15), type boolean, (60000, 15)
  concept[:15, :15] = np.eye(15) # hardcode some entries to 1 # TODO WHY
  fig = Figure(figsize=(3, 3))
  canvas = FigureCanvas(fig)
  axes = fig.gca()
  axes.set_xlim([0, 10])
  axes.set_ylim([0, 10])
  axes.axis('off')
  width, height = fig.get_size_inches() * fig.get_dpi()
  width = int(width)
  height = int(height)
  location = [(1.3, 1.3), (3.3, 1.3), (5.3, 1.3), (7.3, 1.3), (1.3, 3.3),
              (3.3, 3.3), (5.3, 3.3), (7.3, 3.3), (1.3, 5.3), (3.3, 5.3),
              (5.3, 5.3), (7.3, 5.3), (1.3, 7.3), (3.3, 7.3), (5.3, 7.3)] # all possible locations for shape to appear
  x = np.zeros((n_sample, width, height, 3), dtype='uint8') # specify data type to save memory / avoid warning
  color_array = ['green', 'red', 'blue', 'black', 'orange', 'purple', 'yellow']
  if not skip:
    for i in range(n_sample):
      ##############################################
      fig = Figure(figsize=(3, 3))
      canvas = FigureCanvas(fig)
      axes = fig.gca()
      axes.set_xlim([0, 10])
      axes.set_ylim([0, 10])
      axes.axis('off')
      # added by Tony, without reset all x will draw on same canvas, bug
      ##############################################
      location_bool = np.zeros(15) # [Added by Tony, or it will inf loop, bug]: for each image, record whether the location has shape already, to avoid overlap
      if i % 1000 == 0:
        print('{} images are created'.format(i))
      if concept[i, 5] == 1: # consider concept 5, of image i. 1 means that concept needs to appear in the image generated
        a = np.random.randint(15) # select a random int to index into variable "location"
        while location_bool[a] == 1: # if the place has already been taken by other shape
          a = np.random.randint(15) # sample it again # these three lines amount to finding a random place in image that is not taken
        location_bool[a] = 1 # mark that the place chosen is chosen
        axes.plot(
            location[a][0], # x pos
            location[a][1], # y pos
            'x', # shape
            color=color_array[np.random.randint(100) % 7], # a random color
            markersize=8, # size of shape
            mew=4,) # marker edge width
            # ms=8) # redundant argument. bug.
      if concept[i, 6] == 1: # repeat the same thing for concept 6
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '3',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 7] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            's',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 8] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            'p',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 9] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '_',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 10] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            'd',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 11] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            'd',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 12] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            11,
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 13] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            'o',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 14] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '.',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 0] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '+',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 1] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '1',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 2] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '*',
            color=color_array[np.random.randint(100) % 7],
            markersize=30,
            mew=3,)
            # ms=5)
      if concept[i, 3] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            '<',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      if concept[i, 4] == 1:
        a = np.random.randint(15)
        while location_bool[a] == 1:
          a = np.random.randint(15)
        location_bool[a] = 1
        axes.plot(
            location[a][0],
            location[a][1],
            'h',
            color=color_array[np.random.randint(100) % 7],
            markersize=8,
            mew=4,)
            # ms=8)
      canvas.draw()
      image = np.fromstring(
          canvas.tostring_rgb(), dtype='uint8').reshape(width, height, 3)
      x[i, :, :, :] = image
      fig.clf() # they also did not close the graph: their cluster probably has 500G memory so it does not matter :)

      # imgplot = plt.imshow(image)
      # plt.show()

    # create label by booling functions
    y = np.zeros((n_sample, 15))
    y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
    y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
    y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
    y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
    y[:, 4] = concept[:, 1] + concept[:, 4]
    y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
    y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
    y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
    y[:, 8] = concept[:, 2]
    y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
    y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
    y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
    y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
    y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
    y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])

    np.save('x_data.npy', x)
    np.save('y_data.npy', y)
    np.save('concept_data.npy', concept)

  return width, height

def load_xyconcept(n, pretrain):
  """Loads data and create label for toy dataset."""
  concept = np.load('concept_data.npy')
  y = np.zeros((n, 15))
  y[:, 0] = ((1 - concept[:, 0] * concept[:, 2]) + concept[:, 3]) > 0
  y[:, 1] = concept[:, 1] + (concept[:, 2] * concept[:, 3])
  y[:, 2] = (concept[:, 3] * concept[:, 4]) + (concept[:, 1] * concept[:, 2])
  y[:, 3] = np.bitwise_xor(concept[:, 0], concept[:, 1])
  y[:, 4] = concept[:, 1] + concept[:, 4]
  y[:, 5] = (1 - (concept[:, 0] + concept[:, 3] + concept[:, 4])) > 0
  y[:, 6] = np.bitwise_xor(concept[:, 1] * concept[:, 2], concept[:, 4])
  y[:, 7] = concept[:, 0] * concept[:, 4] + concept[:, 1]
  y[:, 8] = concept[:, 2]
  y[:, 9] = np.bitwise_xor(concept[:, 0] + concept[:, 1], concept[:, 3])
  y[:, 10] = (1 - (concept[:, 2] + concept[:, 4])) > 0
  y[:, 11] = concept[:, 0] + concept[:, 3] + concept[:, 4]
  y[:, 12] = np.bitwise_xor(concept[:, 1], concept[:, 2])
  y[:, 13] = (1 - (concept[:, 0] * concept[:, 4] + concept[:, 3])) > 0
  y[:, 14] = np.bitwise_xor(concept[:, 4], concept[:, 3])
  # Tony's note: This is how they generate labels for images. Notice only the first 5 concepts out of 15 is used
  # and the y is 15 dimensional
  if not pretrain:
    x = np.load('x_data.npy') / 255.0
    return x, y, concept
  return 0, y, concept


def main():
  n = 6000  # total dataset size
  n0 = int(n * 0.8)  # training set size
  """Creates the toy dataset main."""
  # Prepares dataset
  print("creating dataset")
  skip = False # True if you have run create_dataset once already and have the npy files in directory
  if skip:
    print("Skipping generation and saving of data, make sure to have x_data.npy, y_data.npy, concept_data.npy in your directory")
  width, height = create_dataset(skip, n_sample=n)
  # Loads dataset
  print("loading dataset")
  x, y, concept = load_xyconcept(n, pretrain=False)
  x_train = x[:n0, :]
  x_val = x[n0:, :]
  y_train = y[:n0, :]
  y_val = y[n0:, :]
  # Loads model
  print("loading model")
  trained = True # change to True if "conv_s13.h5" already exist in directory
  _, _, feature_dense_model = toy_helper.load_model(
      x_train, y_train, x_val, y_val, width=300,
               height=300, channel=3, pretrain=trained)
  print("creating feature")
  toy_helper.create_feature(x, width, height, feature_dense_model)
  # Runs after create_feature
  print("creating cluster")
  toy_helper.create_cluster(concept)

