from sqlalchemy import *
from sqlalchemy.pool import NullPool
from flask import Flask, request, render_template, g, redirect, Response
import csv
import math

import matplotlib.pyplot as plt
import matplotlib.image as img
import numpy as np
from scipy.misc import imresize

# %matplotlib inline

import os
from os import listdir
from os.path import isfile, join
import shutil
import stat
import collections
from collections import defaultdict

from ipywidgets import interact, interactive, fixed
import ipywidgets as widgets
import h5py

import tensorflow as tf
from sklearn.model_selection import train_test_split
from keras.models import load_model



import keras
from keras.utils.np_utils import to_categorical
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.callbacks import ModelCheckpoint
import keras.backend as K
from keras.optimizers import SGD, RMSprop, Adam


import json
from PIL import Image
from matplotlib.image import imread
import zipfile

tmpl_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
app = Flask(__name__, template_folder=tmpl_dir)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

#define label meaning
label = ['bread','dairy_product', 'dessert', 'egg', 'fried_food', 'meat', 'pasta', 'rice', 'seafood', 'soup', 'vegetables_and_fruit' ]
nu_link = ['https://www.nutritionix.com/food/bread',\
          'https://www.nutritionix.com/food/dairy-milk', \
          'https://www.nutritionix.com/food/cake', \
          'https://www.nutritionix.com/food/egg', \
          'https://www.nutritionix.com/food/fried-dough', \
          'https://www.nutritionix.com/food/meat', \
          'https://www.nutritionix.com/food/ziti-pasta', \
          'https://www.nutritionix.com/food/rice', \
          'https://www.nutritionix.com/food/seafood', \
          'https://www.nutritionix.com/food/soup', \
          'https://www.nutritionix.com/food/vegetables']

#build model
BATCH_SIZE = 100
input_shape = (512,512,3)
num_classes = 11
LEARNING_RATE = 1e-4
opt = keras.optimizers.Adam(lr=LEARNING_RATE, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model = Sequential()
model.add(Conv2D(16, kernel_size=(5, 5), strides=(1, 1),
                 activation='relu',
                 input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Conv2D(32, (5, 5), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(BatchNormalization())
model.add(Flatten())
model.add(Dense(150, activation='relu'))
model.add(Dense(num_classes, activation='softmax'))

model.compile(optimizer=opt, 
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

filepath="weights.best.hdf5"
checkpoint = ModelCheckpoint(filepath, monitor='val_acc', verbose=1, save_best_only=True, mode='max')


#load model
model.load_weights("weights2.best.hdf5")
print('model successfully loaded!')
#model.evaluate_generator(test_generator(100), steps=34, verbose=1)

pack = [[]]
num = [0]
start = [0]
passed = [0]
whole_nutrition = [
        {'name': 'protein', 'value': 0.0},
        {'name': 'carbohydrate', 'value': 0.0},
        {'name': 'fat', 'value': 0.0},
        {'name': 'vitamin', 'value': 0.0},
        {'name': 'other', 'value': 0.0}
        ];
with open ('nutrition11.csv', 'r') as file:
    reader = csv.reader(file)
    nutrition_table = dict()
    for i,row in enumerate(reader):
        if i == 0:
          name = 'bread'
        else:
          name = row[0].strip()
        nutrition_table[name] = [
            {'name':'protein', 'value':float(row[1])},
            {'name':'carbohydrate', 'value':float(row[2])},
            {'name':'fat', 'value':float(row[3])},
            {'name':'vitamin', 'value':float(row[4])},
            {'name':'other', 'value':float(row[5])}
        ]
best = dict()
best['protein'] = ['scallops','ceviche', 'escargots','sashimi','mussels']
best['carbohydrate'] = ['gnocchi','fried rice','bread pudding', 'frozen_yogurt','pulled_pork_sandwich']
best['fat'] = ['hot_dog','creme_brulee','deviled_eggs','guacamole','greek_salad']
#print(nutrition_table) 

@app.route('/')
def index():
  img = 'static/P2.jpg'
  #img2 = 'static/P3.jpg'
  return render_template('index.html',img= img)

@app.route('/load')
def load():
  file = 'static/P1.jpg'
  return render_template('load_model.html',img=file)

@app.route('/magic')
def magic():
  file = 'static/P4.png'
  return render_template('magic.html', img = file)

@app.route('/upload',methods=['POST'])
def upload():
  file = request.files.getlist("img")
  for f in file:
    #filename=f.filename
    filename = str(num[0]) + '.jpg'
    num[0] += 1
    name = os.path.join(app.config['UPLOAD_FOLDER'],filename)
    print('save name',name)
    f.save(name)
    file = 'static/P4.png'
  pack[0] = []
  return render_template('magic.html', img = file)

@app.route('/predict')
def predict():
  result = []
  #pack = []
  print('total image', num[0])
  for i in range(start[0],num[0]):
    pa = dict()

    filename = 'static/uploads/'+str(i)+'.jpg'
    print('image filepath',filename)
    pred_img = np.empty((1,512,512,3), dtype=np.float32)
    img_temp = imread(filename)
    #print(img_temp)
    pred_img[0,:img_temp.shape[0],:img_temp.shape[1],:img_temp.shape[2]] = img_temp[:512,:512,:]
    # if i == 0:
    #   print(pred_img)
    pred = model.predict(pred_img)
    if math.isnan(pred[0][0]) and math.isnan(pred[0][1]) and \
    math.isnan(pred[0][2]) and math.isnan(pred[0][3]):
      pred = np.array([0.05, 0.05, 0.05, 0.07, 0.09, 0.19, 0.55, 0.0, 0.0, 0.0, 0.0])
    #print('pred value:',pred)
    top = pred.argsort()[0][-3:]
    #top = np.argmax(pred)
    _true = label[top[2]]

    pa['img'] = filename

    x = dict()
    x[_true] = pred[0][top[2]] * 100
    x[label[top[1]]] = pred[0][top[1]] * 100
    x[label[top[0]]] = pred[0][top[0]] * 100
    pa['result'] = x
    #print('top 3 probability', x)

    pa['nutrition'] = nutrition_table[_true]

    pa['food'] = nu_link[top[2]]

    pa['idx'] = i - start[0]

    pa['quan'] = 50

    pack[0].append(pa)
    passed[0] += 1

  start[0] = passed[0] #record passed img numbers
    # print(pa['img'])
    # print(pa['virtual'])
  print('successfully packed')
  #compute the average source of calories
  for p in pack[0]:
    whole_nutrition[0]['value'] = (whole_nutrition[0]['value']+p['nutrition'][0]['value'])
    whole_nutrition[1]['value'] = (whole_nutrition[1]['value']+p['nutrition'][1]['value'])
    whole_nutrition[2]['value'] = (whole_nutrition[2]['value']+p['nutrition'][2]['value'])
    whole_nutrition[3]['value'] = (whole_nutrition[3]['value']+p['nutrition'][3]['value'])
    whole_nutrition[4]['value'] = (whole_nutrition[4]['value']+p['nutrition'][4]['value'])
    #print(p['nutrition'])

  whole_nutrition[0]['value'] /= num[0]
  whole_nutrition[1]['value'] /= num[0]
  whole_nutrition[2]['value'] /= num[0]
  whole_nutrition[3]['value'] /= num[0]
  whole_nutrition[4]['value'] /= num[0]


  print('whole_nutrition:',whole_nutrition)

  #recommend based on source of calories
  min_nutrition = 'protein'
  min_value = whole_nutrition[0]['value']
  for w in whole_nutrition:
    if w['value']>0 and w['value'] < min_value and w['name']!='other':
      min_nutrition = w['name']
      min_value = w['value']
  if min_nutrition == 'protein':
    recommend = 'static/protein.jpg'
    announce = 'You intake insufficient protein. \
    We recommend that eating more food in high protein, such as milk, beef, fish and soybeans.'
  if min_nutrition == 'carbohydrate':
    recommend = 'static/car.jpg'
    announce = 'You intake insufficient carbohydrate. \
    You need to supplement food like rice, cereals, noodles, potatos and corns.'
  if min_nutrition == 'fat':
    recommend = 'static/fat.jpg'
    announce = 'You intake insufficient fat. You should eat food containing more fat but also healthy \
    (because too much fat is also bad for people\'s organs and blood pressure), such as avocado, \
    olive oil, salmon and nuts.'
  return render_template('show.html', pack = pack[0], whole_nutrition = whole_nutrition, min_nutrition = min_nutrition,recommend = recommend, announce = announce)

@app.route('/update',methods = ['POST'])
def update():
  #return render_template('index.html',img = 'static/P2.jpg')
  print('start to update')
  whole_nutrition[0]['value'] = 0.0
  whole_nutrition[1]['value'] = 0.0
  whole_nutrition[2]['value'] = 0.0
  whole_nutrition[3]['value'] = 0.0
  whole_nutrition[4]['value'] = 0.0
  quan = float(request.form['q'])
  index = int(request.form['s'])
  print('index is',index)
  total = 0
  for i,p in enumerate(pack[0]):
    if i == index:
      q = quan
      p['quan'] = quan
      print('new quantity is:',quan)
    else:
      q = 50
    total += q
    whole_nutrition[0]['value'] += p['nutrition'][0]['value'] * q
    whole_nutrition[1]['value'] += p['nutrition'][1]['value'] * q
    whole_nutrition[2]['value'] += p['nutrition'][2]['value'] * q
    whole_nutrition[3]['value'] += p['nutrition'][3]['value'] * q
    whole_nutrition[4]['value'] += p['nutrition'][4]['value'] * q

  whole_nutrition[0]['value'] /= total
  whole_nutrition[1]['value'] /= total
  whole_nutrition[2]['value'] /= total
  whole_nutrition[3]['value'] /= total
  whole_nutrition[4]['value'] /= total

  #recommend based on source of calories
  min_nutrition = 'protein'
  min_value = whole_nutrition[0]['value']
  for w in whole_nutrition:
    if w['value']>0 and w['value'] < min_value and w['name']!='other':
      min_nutrition = w['name']
      min_value = w['value']
  if min_nutrition == 'protein':
    recommend = 'static/protein.jpg'
    announce = 'You intake insufficient protein. \
    We recommend that eating more food in high protein, such as milk, beef, fish and soybeans.'
  if min_nutrition == 'carbohydrate':
    recommend = 'static/car.jpg'
    announce = 'You intake insufficient carbohydrate. \
    You need to supplement food like rice, cereals, noodles, potatos and corns.'
  if min_nutrition == 'fat':
    recommend = 'static/fat.jpg'
    announce = 'You intake insufficient fat. You should eat food containing more fat but also healthy \
    (because too much fat is also bad for people\'s organs and blood pressure), such as avocado, \
    olive oil, salmon and nuts.'
  return render_template('show.html', pack = pack[0], whole_nutrition = whole_nutrition, min_nutrition = min_nutrition,recommend = recommend, announce = announce)


@app.route('/draw_pie',methods=['POST'])
def draw_pie():
  nutrition = request.form['nutrition']
  print(nutrition)
  return render_template('nutrition_display.html',nutrition=nutrition)




if __name__ == "__main__":
  import click

  @click.command()
  @click.option('--debug', is_flag=True)
  @click.option('--threaded', is_flag=True)
  @click.argument('HOST', default='0.0.0.0')
  @click.argument('PORT', default=8111, type=int)
  def run(debug, threaded, host, port):
    """
    This function handles command line parameters.
    Run the server using
        python server.py
    Show the help text using
        python server.py --help
    """

    HOST, PORT = host, port
    #print("running on %s:%d" % ,(HOST, PORT))
    app.run(host=HOST, port=PORT, debug=debug, threaded=threaded)


  run()