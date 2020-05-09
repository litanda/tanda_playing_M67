#!/usr/bin/env python
# coding: utf-8

# In[1]:


from __future__ import absolute_import, division, print_function, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
import os
import glob
from matplotlib import gridspec


try:
  # %tensorflow_version only exists in Colab.
  get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
  pass
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

from tensorflow.keras import regularizers
from tensorflow.keras.models import load_model


# In[ ]:


def import_data(dr = None, condition = None):
    all_files = glob.glob(rootdr + condition)

    li = []

    for filename in all_files:
        df = pd.read_csv(filename, index_col=0, header=0)
        li.append(df)

    data_raw = pd.concat(li, axis=0, ignore_index=True)

    data = data_raw.copy()

    data = data.ix[data['evol_stage'] >= 1]

    log_column = ['star_age','effective_T','luminosity']
    data[log_column] = np.log10(data[log_column])
    #data.tail()
    return data


# In[ ]:


def prepare_data_for_one_mode(data = None, mode_column_name = None):
    data_new = data.copy()
    data_new[mode_column_name] = np.log10(data_new[mode_column_name])
    data_new = data_new.replace([np.inf, -np.inf], np.nan)
    data_new.isna().sum()
    data_new = data_new.dropna()
    data_new = data_new.sample(frac = 1.0, random_state=1)
    return data_new


# In[ ]:


def prepare_train_set(data = None, inputs = None, outputs = None):
    train_dataset = data[inputs]
    train_labels = data[outputs]
    return train_dataset, train_labels


# In[ ]:


def plot_history(history, savefig = None):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('log(Mean Abs Error [MPG])')
    plt.plot(hist['epoch'], np.log10(hist['mae']),
             label='Train Error')
    plt.plot(hist['epoch'], np.log10(hist['val_mae']),
             label = 'Val Error')
    #plt.ylim([-8,-1])
    plt.legend()
    plt.savefig(savefig + '.png')
    plt.close()
    print('figure is saved in' + savefig)
    return



# In[ ]:


def tf_model_one_mode(mode_name = None, train_dataset = None, train_labels = None, epochs = None, savemodel = None):
    model = tf.keras.Sequential()
    model.add(layers.Dense(128, activation ='elu', input_shape=[len(train_dataset.keys())],kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(128, activation ='elu',kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(128, activation ='elu',kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(128, activation ='elu',kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(128, activation ='elu',kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(128, activation ='elu',kernel_regularizer=regularizers.l2(0.000001)))
    model.add(layers.Dense(1, activation ='linear'))

    #optimizer = tf.keras.optimizers.RMSprop(0.001)
    #optimizer = tf.keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
    optimizer = tf.keras.optimizers.Nadam(learning_rate=0.001, beta_1=0.9995, beta_2=0.9999)

    model.compile(loss='mae',optimizer=optimizer,metrics=['mae', 'mse'])
#Use the .summary method to print a simple description of the model
#train model

# Display training progress by printing a single dot for each completed epoch
    class PrintDot(keras.callbacks.Callback):
      def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')

    #EPOCHS = 1000
    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', min_delta=0.000001, patience = 0.3*epochs)

    EPOCHS = epochs

    history = model.fit(train_dataset, train_labels, epochs=EPOCHS,
                        batch_size= int(0.6*len(train_dataset)),
                        validation_split = 0.4, verbose=0, callbacks=[early_stop,])# PrintDot()])#callbacks = [PrintDot()])

    model.save(savemodel)

    plot_history(history, savefig = savemodel)

    loss, mae, mse = model.evaluate(train_dataset, train_labels, verbose=2)
    print("Testing set Mean Abs Error for" + mode_name +": {:5.2f}".format(mae))

    return model


# In[ ]:


def plot_mode_predictions(model = None,mode_column_name=None, train_dataset = None, train_labels = None, savefig = None):
    print(model.summary())
    loss, mae, mse = model.evaluate(train_dataset, train_labels, verbose=2)
    predictions = model.predict(train_dataset)

    error_nu = 10**predictions[:,0] - 10**train_labels[mode_column_name].values

    plt.figure(figsize=(10, 6))
    gs = gridspec.GridSpec(1, 2) #, width_ratios=[2, 1]) #,height_ratios=[2,1])
    p0 = plt.subplot(gs[0,0])
    p0.axis([4.5, 3.5, 3.0, -0.5 ])
    plt.xlabel(r'$T_{\rm eff}$ (K)')
    plt.ylabel(r'($\nu$ ($\mu$Hz)')
    p0.scatter(train_dataset['effective_T'], train_labels[mode_column_name], marker='.',color = 'k') #, facecolors='none', edgecolors='k')
    p0.scatter(train_dataset['effective_T'].values, predictions[:,0], marker='.',color = 'r') #, facecolors='none', edgecolors='k')

    p1 = plt.subplot(gs[0,1])
    p1.hist(error_nu, bins = 200,range = [-0.02,0.02])
    plt.xlabel("Prediction Error [nu_0_1] (microHz)")
    plt.ylabel("Count")
    plt.savefig(savefig)
    print('Fig saved in' + savefig)
    return predictions


# In[8]:


get_ipython().system("ls '/Users/litz/Documents/GitHub/data/'")
rootdr = '/Users/litz/Documents/GitHub/data/'


# In[10]:


data = import_data(dr = rootdr, condition = 'grid2_subset/simple_grid_v3_s1/*.csv')
data.tail()


# In[ ]:


mode_n_orders = np.arange(1,41)
mode_column_names = [f'nu_0_{i}' for i in mode_n_orders]
mode_column_names = ['nu_0_1']


# In[ ]:


for mode_column_name in mode_column_names:

    data_new = prepare_data_for_one_mode(data = data, mode_column_name = mode_column_name)

    print(len(data_new['model_number']))

    inputs = ['initial_mass','initial_Yinit','initial_feh','initial_MLT','initial_fov',
          'star_mass','star_age','effective_T','log_g','luminosity']
    outputs = [mode_column_name]
    train_dataset, train_labels = prepare_train_set(data = data_new, inputs = inputs, outputs = outputs)

    print("tf is training for " + mode_column_name)

    model = tf_model_one_mode(mode_name = mode_column_name,
                              train_dataset = train_dataset,
                              train_labels = train_labels,
                              epochs = 100000,
                              savemodel = rootdr + "grid_2_NN" + mode_column_name + ".h5")

    predictions = plot_mode_predictions(model = model,
                                        mode_column_name = mode_column_name,
                                        train_dataset = train_dataset,
                                        train_labels = train_labels,
                                        savefig = rootdr + mode_column_name + 'on_grid.png')



# In[15]:

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:
