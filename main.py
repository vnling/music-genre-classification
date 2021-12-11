import tfidf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import pandas as pd
import numpy as np
import pos
import nltk
import word_count

file = 'lyrics.csv'
df = pd.read_csv(file)

tokenizer = nltk.RegexpTokenizer(r"\w+")
for idx, row in df.iterrows():
    lyrics = df.loc[idx,'lyrics']
    clean_lyrics = tokenizer.tokenize(lyrics)
    clean_lyrics = " ".join(clean_lyrics)
    df.loc[idx,'lyrics'] = clean_lyrics

tf_idf = tfidf.get_tf_idf(df.lyrics)
pos_tags = pos.get_pos(df.lyrics)
# lyric_word_count = word_count.get_word_count(df.lyrics)

features = [tf_idf.rename('tfidf'), pos_tags]
X = pd.concat(features, axis=1)
X = np.array(X)

df.genre = pd.Categorical(df.genre)
df.genre = df.genre.cat.codes
y = np.array(df['genre'])

Xtr, Xts, ytr, yts = train_test_split(X,y,test_size=0.3)


# Linear Regression with Cross validation
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import precision_recall_fscore_support

# nfold = 10
# kf = KFold(n_splits=nfold)
# acc = np.zeros(nfold)
# prec = np.zeros(nfold)
# rec = np.zeros(nfold)
# f1 = np.zeros(nfold)

acc = np.zeros(5)

# Finding the best penalty
for i,p in enumerate([0.0001, 0.001, 0.01, 0.1, 1.0]):

    # Scale the data
    scal = StandardScaler()
    Xtr1 = scal.fit_transform(Xtr)
    Xts1 = scal.transform(Xts)    
    
    # Fit a model    
    logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty='l2', C=p)
    logreg.fit(Xtr1, ytr)

    #  Predict on test samples and measure accuracy
    yhat = logreg.predict(Xts1)
    acc[i] = np.mean(yhat == yts)


print(acc)

# for i, I in enumerate(kf.split(X)):
    
#     # Get training and test data
#     train, test = I
#     Xtr = X[train,:]
#     ytr = y[train]
#     Xts = X[test,:]
#     yts = y[test]
    
#     # Scale the data
#     scal = StandardScaler()
#     Xtr1 = scal.fit_transform(Xtr)
#     Xts1 = scal.transform(Xts)    
    
#     # Fit a model    
#     reg.fit(Xtr1, ytr)
    
#     # Predict on test samples and measure accuracy
#     yhat = reg.predict(Xts1)
#     acc[i] = np.mean(yhat == yts)
    
#     # Measure other performance metrics
#     prec[i],rec[i],f1[i],_  = precision_recall_fscore_support(yts,yhat,average='') 
    

# # Take average values of the metrics
# precm = np.mean(prec)
# recm = np.mean(rec)
# f1m = np.mean(f1)
# accm= np.mean(acc)

# # Compute the standard errors
# prec_se = np.std(prec)/np.sqrt(nfold-1)
# rec_se = np.std(rec)/np.sqrt(nfold-1)
# f1_se = np.std(f1)/np.sqrt(nfold-1)
# acc_se = np.std(acc)/np.sqrt(nfold-1)

# print('Precision = {0:.4f}, SE={1:.4f}'.format(precm,prec_se))
# print('Recall =    {0:.4f}, SE={1:.4f}'.format(recm, rec_se))
# print('f1 =        {0:.4f}, SE={1:.4f}'.format(f1m, f1_se))
# print('Accuracy =  {0:.4f}, SE={1:.4f}'.format(accm, acc_se))

# logreg = LogisticRegression(multi_class='multinomial', solver='lbfgs', penalty="l2", C=0.5,)
# logreg.fit(Xtr,ytr)
# yhat = logreg.predict(Xts)
# print('Accuracy of logistic regression classifier on test set: {:.2f}'.format(np.mean(yhat == yts)))

# from sklearn import svm
# clf = svm.SVC(kernel='linear', C=0.5, decision_function_shape='ovo')
# clf.fit(Xtr,ytr)
# yhat = clf.predict(Xts)
# print('Accuracy of SVM on test set: {:.2f}'.format(np.mean(yhat == yts)))

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense

# import tensorflow.keras.backend as K
# K.clear_session()

# nin = X.shape[1]
# nh = 300
# nout = np.max(y) + 1
# model = Sequential()
# model.add(Dense(units=nh, input_shape=(nin,), activation='sigmoid',name='hidden'))
# model.add(Dense(units=nout, activation='softmax', name='output'))

# from tensorflow.keras import optimizers

# opt = optimizers.Adam(learning_rate=1e-3) 
# model.compile(optimizer=opt,
#               loss='sparse_categorical_crossentropy',
#               metrics=['accuracy'])

# hist = model.fit(Xtr, ytr, epochs=1000, batch_size=100, validation_data=(Xts,yts), verbose=False)
# print(hist.history['val_accuracy'])

# tr_accuracy = hist.history['accuracy']
# val_accuracy = hist.history['val_accuracy']