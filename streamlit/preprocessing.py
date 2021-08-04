from tensorflow.keras.preprocessing.image import load_img,img_to_array
import cv2
import numpy as np

def preprocess(imgpath):
    img=cv2.imread(imgpath)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
    img_black=cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    mask=img_black>5
    temp=img[np.ix_(mask.any(1),mask.any(0))]
    if temp.shape[0]==0:
        clean_img=img
    else:
        img0=img[:,:,0][np.ix_(mask.any(1),mask.any(0))]
        img1=img[:,:,1][np.ix_(mask.any(1),mask.any(0))]
        img2=img[:,:,2][np.ix_(mask.any(1),mask.any(0))]
        clean_img=np.stack([img0,img1,img2],axis=-1)
    image_final=cv2.addWeighted(clean_img,0.8,cv2.GaussianBlur(clean_img,(15,15),30),0.2,80)
    img=cv2.resize(image_final,(900,900))
    img=img_to_array(img)/255.0
    img=np.expand_dims(img,axis=0)
    return img