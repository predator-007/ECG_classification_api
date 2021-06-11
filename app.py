from flask import Flask,jsonify
from flask import Flask,request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
import os
from flask_cors import CORS
import tensorflow as tf
graph=tf.compat.v1.get_default_graph()
app = Flask(__name__)
CORS(app)

index = ['Fusion Beat','Normal Beat','Unknown Beat','Supraventricular ectopic Beat','Ventricular ectopic beat']
        

@app.route('/')
def working():
    return jsonify("ECG arrhythmia classification api working")
    

@app.route('/predict',methods=["POST"])
def predict_class():
    model=load_model("ecg_model/my_ecg_model")
    imagef=request.files['image']
    imagen=imagef.filename
    imagef.save(os.path.join(os.getcwd(), imagen))
    img=image.load_img(imagen,target_size=(64,64))
    x=np.array(img)
    x=np.expand_dims(x,axis=0)
    resp=model.predict(x)
    print(resp)    
    resi=np.argmax(resp)
    res=index[resi]
    print("predictions",res)
    return jsonify(res)
    
if __name__=="__main__":
    app.run(debug=True,port=5000)  

