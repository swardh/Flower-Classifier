import tensorflow as tf
import numpy as np
from PIL import Image
import streamlit as st

model = tf.keras.models.load_model('inceptV39295.h5')

classes = ['Daisy', 'Dandelion', 'Rose', 'Sunflower', 'Tulip']

def prediction(img):
    image = Image.open(img)
    resized = image.resize((224, 224))
    npImg = np.array(resized)
    rescaled = npImg / 255
    imgage = rescaled.reshape(1,224,224,3)

    probs = model.predict(imgage)

    st.write(f'Prediction : {classes[np.argmax(probs)]}')
    st.write('')
    st.write('(Resized image)')
    st.image(resized)