import tfidf
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

file = 'lyrics.csv'
df = pd.read_csv(file)
tf_idf = tfidf.get_tf_idf(file)

X = tf_idf
X = X.reshape((X.shape[0],1))
df.genre = pd.Categorical(df.genre)
df.genre = df.genre.cat.codes
y = np.array(df['genre'])

Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=0.3)

from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty="l2", C=0.5)
logreg.fit(Xtr,ytr)
yhat = logreg.predict(Xts)
print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(np.mean(yhat == yts)))

from sklearn import svm
clf = svm.SVC(kernel='linear', C=1, decision_function_shape='ovo')
clf.fit(Xtr,ytr)
yhat = clf.predict(Xts)
print('Accuracy of SVM on test set: {:.2f}'.format(np.mean(yhat == yts)))

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

import tensorflow.keras.backend as K
K.clear_session()

nin = X.shape[1]
nh = 100
nout = np.max(y) + 1
model = Sequential()
model.add(Dense(units=nh, input_shape=(nin,), activation='sigmoid',name='hidden'))
model.add(Dense(units=nout, activation='softmax', name='output'))

from tensorflow.keras import optimizers

opt = optimizers.Adam(learning_rate=1e-3) 
model.compile(optimizer=opt,
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

hist = model.fit(Xtr, ytr, epochs=1000, batch_size=30, validation_data=(Xts,yts))

tr_accuracy = hist.history['accuracy']
val_accuracy = hist.history['val_accuracy']