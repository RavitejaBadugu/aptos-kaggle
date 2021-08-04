import streamlit as st
import requests
import json
import shutil
import os
import numpy as np
from preprocessing import preprocess

URL='http://model_server:8000/v1/models/densenet:predict'

st.title('Welcome to this app')
st.subheader('In this app we predict the chances of diabetic retinopathy')
st.write('Basically it the kaggle competition task where I build the model.')
st.write('This is the app to prototype the production')

st.markdown('** Below please upload the retina images taken using fundus photography **')

uploaded_img=st.file_uploader('upload the image',type=['jpg','png'])

if uploaded_img is not None:
    img_name=uploaded_img.name
    with open(f'tmp/{img_name}','wb') as f:
        shutil.copyfileobj(uploaded_img, f)
    img=preprocess(f'tmp/{img_name}')
    data=json.dumps({"signature_name": "serving_default",'instances':img.tolist()})
    headers={"content-type": "application/json"}
    response=requests.post(URL,data=data,headers=headers)
    result=json.loads(response.text)['predictions'][0]
    os.remove(f'tmp/{img_name}')
    max_index=np.argmax(np.array(result))
    dis=['No DR','Mild','Moderate','Severe','Proliferative DR']
    if result[max_index]>0.85:
        st.success(f"model predicts having {dis[max_index]} diabetic retinopathy with probabilit of  {result[max_index]}")
    else:
        st.success('model is not so confident about the predictions')
    