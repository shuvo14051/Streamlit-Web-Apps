import streamlit as st
from keras.models import load_model
from PIL import Image
from util import classify

st.title('Pneumonia classification')

st.header('Please upload an image of a chest X-ray')

file = st.file_uploader('', type=['jpeg', 'png', 'jpg'])

# load classifier
model = load_model('pneumonia.h5')

# load class labels
class_names = ['pneumonia', 'normal']

# display image
if file is not None:
    image = Image.open(file).convert('RGB')
    st.image(image, use_column_width=True)

    # classifiy image
    class_name, conf_score = classify(image, model, class_names)

    st.write("## {}".format(class_name))
    st.write("### {}".format(conf_score))