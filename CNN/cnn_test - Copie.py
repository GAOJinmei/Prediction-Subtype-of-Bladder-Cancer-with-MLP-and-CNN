import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn import preprocessing
from keras.utils import np_utils
from sklearn.model_selection import train_test_split
from sklearn.metrics import multilabel_confusion_matrix, confusion_matrix, ConfusionMatrixDisplay

import tensorflow as tf
from keras import layers
from sklearn.datasets import make_blobs
# from tensorflow.keras.optimizers import SGD, Adam
from keras.models import Sequential
from keras.layers import Input, Dense, Activation
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from keras.layers import LSTM,Dense, Flatten
from keras.models import Sequential
from keras.layers import Convolution1D
from keras.layers import MaxPooling1D
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Dropout
from tensorflow.keras.layers import BatchNormalization
from keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from keras.layers import Input, Dense, Activation,GaussianNoise



#df1 = pd.read_csv('D:/ROSALIND_problems/Deep_learning_project/df_transpose.csv',sep=';')
#df1.rename(columns = {"Unnamed: 0":"barcode"},inplace = True)
#df1= df1.drop(columns='barcode')


# Read gene
df1 = pd.read_csv('D:/ROSALIND_problems/Deep_learning_project/gene_data_ordered.csv',sep=';')
df1.head()
df1.shape

print('there is {} patient/observation/row for {} gene/feature/columns , no extensive quality control yet'.format(df1.shape[0],df1.shape[1]))


# delete the columns that contains all the same value
X1 = df1[[c for c in list(df1)
       if len(df1[c].unique())>1]]
len(X1.columns)
X1.head(n=10)

 #drop the coloumns which contains too many NAN if we have many NANs
X2 = X1.dropna(thresh= 10, axis=1)
len(X2.columns)
X2 = X1.fillna(X1.mean())
X2.isna().sum()





###  Read Patient description ###
df2 = pd.read_csv('D:/ROSALIND_problems/Deep_learning_project/data_patient.csv',sep=';')
df2.head()

# choose diagnosis
Y = df2[['primary_diagnosis']]
Y
# check number of cancer type
print('distribution of data is' , Y["primary_diagnosis"].value_counts())
# ploting
number=['358','66','1','1','1']
counts = Y['primary_diagnosis'].value_counts().rename_axis('Cancer_type').reset_index(name='count')
ax = sns.barplot(x='Cancer_type', y='count', data=counts)
# ax.bar_label(ax.containers[0])


#### ENCODING categorical data to integer ( 0 to 5 ) then put into binary  ###
# set 1 to 5 for each cancer types
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
label_encoder = LabelEncoder()
Y["classes"] = label_encoder.fit_transform(Y)
Y
# put into binary
classes = Y["classes"]
labels = np_utils.to_categorical(classes)
print(labels[:10])


#### normalization or scaling of features (genes)  ####
# density plot before normalisation :
sns.set_style("whitegrid")
ax = sns.displot(data=X1, kind='kde', fill=True, height=5, aspect=2)
# Here you can define the x limit
'''
ax.set(xlim=(-50,100))
ax.set(xlabel = 'Gene expression', ylabel = 'Probability Density')
ax.fig.suptitle("Gene expression Distribution",
                fontsize=20, fontdict={"weight": "bold"})
plt.show()
'''

# check range of distribution
print(X1.describe().T.head())
scaler = preprocessing.StandardScaler().fit(X1)
data = scaler.transform(X1)
# log2 scaler
#data = np.log2(X1)
#print(data)
data.shape


#Data spliting

scaler = preprocessing.StandardScaler().fit(X1)
data = scaler.transform(X1)

validation_ratio = 0.30

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.3, random_state=42)
#test_size = validation_ratio)
shape= x_train.shape[1]


print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# weight
from sklearn.utils import class_weight

class_weights = class_weight.compute_class_weight('balanced',
                                                 np.unique(y_train),
                                                 y_train)


class_weights = compute_class_weight(
                                        class_weight = "balanced",
                                        classes = np.unique(y_train),
                                        y = y_train
                                    )
class_weights = dict(zip(np.unique(train_classes), class_weights))
class_weights


#Convert data into 3d tensor
X_train = x_train.reshape(x_train.shape[0], x_train.shape[1], 1)
X_test = x_test.reshape(x_test.shape[0],x_test.shape[1], 1)
print(X_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# set random seed for reproductibility
seed = 7
np.random.seed(seed)




# function to create the model for Keras wrapper to scikit learn
# we will optimize the type of pooling layer (max or average) and the activation function of the 2nd and 3rd convolution layers
def create_cnn_model():
    loss = 'categorical_crossentropy'  # https://keras.io/optimizers
    optimizer = tf.keras.optimizers.SGD(learning_rate=0.001)  # https://keras.io/losses
    metrics = ['accuracy']
    pool_type = 'max'
    dropout_rate = 0
    # create model
    model = Sequential()

    # first layer : convolution
    model.add(Convolution1D(filters = 8 , kernel_size=11, activation='relu', input_shape=(17072,1)))
    model.add(Activation('relu'))
    lambda_l1 = 0.001


    # max pooling and dropout rate if necessary (dropout not null )
    if pool_type == 'max':
        model.add(MaxPooling1D(strides=5))
    if pool_type == 'average':
        model.add(AveragePooling1D(strides=5))
    if dropout_rate != 0:
        model.add(Dropout(rate=dropout_rate))
    #model.add(BatchNormalization())
    model.add(GaussianNoise(0.1))


    # Classification Layer
    model.add(Flatten())
    model.add(Dense(10,activation='relu',kernel_regularizer=regularizers.l2(lambda_l1)))
    #model.add(GaussianNoise(0.1))
    model.add(Dense(7,activation='relu',kernel_regularizer=regularizers.l1(lambda_l1)))
    #model.add(GaussianNoise(0.1))

    model.add(Dense(len(Y['classes'].unique()),activation='Softmax')) #kernel_regularizer=regularizers.l1(lambda_l1)))

    model.compile(optimizer = optimizer, loss = loss, metrics = metrics)





    return model



cnn = create_cnn_model()




cnn.compile(loss = "categorical_crossentropy",
            metrics=['accuracy'],
            optimizer = tf.keras.optimizers.SGD(learning_rate=0.001))
cnn.summary()


#Train the default model

# early_stop = EarlyStopping(monitor = 'categorical_accuracy', mode = 'max', patience=5, restore_best_weights=True)
early_stop = [EarlyStopping(monitor='val_loss', patience=5, verbose=0,min_delta=0.01,restore_best_weights=False)]

history = cnn.fit(X_train,y_train,
                  batch_size=150,
                  epochs=150,
                  callbacks=[early_stop],
                  validation_data=(X_test, y_test))




from sklearn.model_selection import GridSearchCV
from keras.wrappers.scikit_learn import KerasClassifier

model = KerasClassifier(build_fn=create_cnn_model, verbose=1)

size_batch = [100]
nb_epochs = [10]
Gaussian_Noise=[0.1]
param_grid = dict(batch_size=size_batch, epochs=nb_epochs,Gaussian_Noise=Gaussian_Noise)
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=1, cv=3)
grid_result = grid.fit(data, labels)
# summarize result

# summarize results
print("Best: %f using %s" % (grid_result.best_score_, grid_result.best_params_))
means = grid_result.cv_results_['mean_test_score']
stds = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print("%f (%f) with: %r" % (mean, stdev, param))


def display_cv_results(search_results):
    print('Best score = {:.4f} using {}'.format(search_results.best_score_, search_results.best_params_))
    means = search_results.cv_results_['mean_test_score']
    stds = search_results.cv_results_['std_test_score']
    params = search_results.cv_results_['params']
    for mean, stdev, param in zip(means, stds, params):
        print('mean test accuracy +/- std = {:.4f} +/- {:.4f} with: {}'.format(mean, stdev, param))

display_cv_results(grid_result)


        ##  grid  search
from keras.wrappers.scikit_learn import KerasClassifier
# create model
n_epochs_cv=10
# define parameters and values for grid search
param_grid = {
    'pool_type': ['max', 'average'],
    'conv_activation': ['sigmoid', 'tanh'],
    'epochs': [n_epochs_cv],
}


from sklearn.model_selection import GridSearchCV
from keras.utils import to_categorical


n_cv = 3
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=n_cv)
grid_result = grid.fit(X_train, (y_train))






### plot history ###
print (history.history.keys())
# plot loss
loss= history.history['loss']
val_loss= history.history['val_loss']
epochs=range(1,len(loss)+1)
plt.plot(epochs,loss,'y',label='Training loss')
plt.plot(epochs,val_loss,'r',label= 'Testing loss')
plt.title('Training and Testing loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

# plot accuracy
accuracy= history.history['accuracy']
val_accuracy= history.history['val_accuracy']
epochs=range(1,len(accuracy)+1)
plt.plot(epochs,accuracy,'y',label='Training accuracy')
plt.plot(epochs,val_accuracy,'r',label= 'Testing accuracy')
plt.title('Training and Testing accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()


