import streamlit as st
import predict as pred

st.title('Flower image classification with tensorflow')
st.header('This is a flower classifier trained in tensorflow with 92.95 % accuracy')
st.subheader('')
st.text('Upload a picture of a Rose, Sunflower, Tulip, Daisy or Dandelion')

uploaded_file = st.file_uploader('upload flower picture', type=['jpg', 'jpeg', 'png'])
if uploaded_file is not None:

    st.write('(Orginal image)')
    st.image(uploaded_file)
    pred.prediction(uploaded_file)
