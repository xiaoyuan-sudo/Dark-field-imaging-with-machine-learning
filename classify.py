# import the necessary packages
from keras.models import load_model
import numpy as np


testdata = np.load('testdata.npy')
testlabel = np.load('testlabel.npy')

# load the trained convolutional neural network and the label
# binarizer
print("[INFO] loading network...")
model = load_model('pokedex.model')
#lb = pickle.loads(open('lb.pickle', "rb").read())

# classify the input image
print("[INFO] classifying image...")
proba = model.predict(testdata)
#idx = np.argmax(proba)
np.save('proba.npy',proba)


