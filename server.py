from flask import Flask,jsonify
from flask import Flask,request
from keras.models import load_model
from keras.preprocessing import image
from keras_preprocessing.image.utils import img_to_array
import numpy as np
import tensorflow as tf
import cv2
import os
from PIL import Image

app = Flask(__name__)
model=load_model("model.h5")
index = ['Fusion Beat','Normal Beat','Unknown Beat','Supraventricular ectopic Beat','Ventricular ectopic beat']
        

@app.route('/')
def working():
    return jsonify("ECG arrhythmia classification api working")

@app.route('/predict',methods=["POST"])
def predict_class():
    imagef=request.files['file']
    imagen=imagef.filename
    imagef.save(os.path.join(os.getcwd(), imagen))
    img=image.load_img(imagen,target_size=(64,64))
    x=np.array(img)
    x=np.expand_dims(x,axis=0)
    pred=model.predict_classes(x)
    return jsonify(index[pred[0]]) 
    
if __name__=="__main__":
    app.run(debug=True)
