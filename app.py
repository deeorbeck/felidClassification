import streamlit as st
from fastai.vision.all import *
import plotly.express as px
import pathlib
import platform
plt = platform.system()
if plt == 'Linux': pathlib.WindowsPath = pathlib.PosixPath
if plt == "Windows":
    temp = pathlib.PosixPath
    pathlib.PosixPath = pathlib.WindowsPath



# title
st.title("Felid Classification ⭐")

# upload the image
file = st.file_uploader("Attach the image ", type=['png', 'jpg', 'jpeg', 'gif', 'svg', 'webp'])
if file:
    # convet PILImage
    img = PILImage.create(file)
    

    # import model
    model = load_learner(fname="felid_model.pkl")

    # prediction
    pred, pred_id, probs = model.predict(img)
    st.image(img, caption=f'{pred}'.upper())
    st.success(f'Prediction: {pred}')
    st.info(f'Probability: {probs[pred_id]*100:.1f}%')

    #plotting
    fig = px.bar(x = probs * 100, y=model.dls.vocab)
    st.plotly_chart(fig)
