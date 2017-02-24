#from __future__ import division, print_function, absolute_import
import tflearn
import numpy as np
from random import shuffle
import os
import librosa


def extract_mfcc(classes, source):
    features = []
    labels = []
    path = source + '/'
    files = os.listdir(path)
    length = []
    shuffle(files)
    for wav in files:
        # wave = waveform
        # sr = sampling rate
        wave, sr = librosa.load(path + wav, mono=True)
        label = [0] * classes
        label[int(wav[0])] = 1
        labels.append(label)

        mfcc = librosa.feature.mfcc(wave, sr)
        length.append(mfcc.shape[1])
        mfcc = np.pad(mfcc, ((0, 0), (0, 80 - len(mfcc[0]))), mode='constant', constant_values=0)


        features.append(np.array(mfcc))
    print(max(length))
    return features, labels

def train(epochs, saved_model=None):
    if saved_model is not None:
        model.load(saved_model)
    model.fit(trainX, trainY, n_epoch=epochs, validation_set=(testX, testY), show_metric=True, batch_size=batch_size)
    model.save("speechNN.model")

def evaluate(saved_model):
    model.load(saved_model)
    prediction = model.predict(testX)
    acc = 0.0
    conf = [[0]*classes for i in range(classes)]
    for i in range(len(prediction)):
        _y = np.argmax(prediction[i])
        y = np.argmax(testY[i])
        conf[y][_y] += 1
        if _y == y:
            acc += 1.0
    print('Accuracy: {}'.format(acc/len(prediction)))
    print()

    print("\t" +"\t".join([str(i) for i in range(classes)]))
    for i in range(classes):
        s = str(i)
        for j in range(classes):
            s += "\t{}".format(conf[i][j])
        print(s)




learning_rate = 0.0001
batch_size = 30

width = 20  # mfcc features
height = 80  # (max) length of utterance
classes = 10  # digits


trainX, trainY = extract_mfcc(10,'train')
testX, testY = extract_mfcc(10,'test')


# Network building
net = tflearn.input_data([None, width, height])
net = tflearn.lstm(net, 128, dropout=0.8)
net = tflearn.fully_connected(net, classes, activation='softmax')
net = tflearn.regression(net, optimizer='adam', learning_rate=learning_rate, loss='categorical_crossentropy')

model = tflearn.DNN(net, tensorboard_verbose=0)





#train(100, saved_model="speechNN.model")
evaluate('speechNN.model')