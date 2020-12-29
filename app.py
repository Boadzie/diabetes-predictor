import pickle
from fastapi import FastAPI
from starlette.requests import Request
from fastapi.templating import Jinja2Templates
import numpy as np


app = FastAPI()

templates = Jinja2Templates(directory="template")


# the filename of the saved model
filename = 'diabetes.sav'


# load the saved model
loaded_model = pickle.load(open(filename, 'rb'))

@app.get("/")
def home(request: Request): 
    return templates.TemplateResponse("index.html", {"request": request})

@app.post('/predict')
async def predict(request: Request):
    result = {}
    if request.method == "POST":
        # get the features to predict
        form = await request.form()
        # form data
        age = form["Age"]
        glucose = form["Glucose"]
        bmi = form["BMI"]
        # create the features list for prediction
        features_list = [int(glucose), int(bmi), int(age)]

        # get the prediction class
        prediction = loaded_model.predict([features_list])

        # get the prediction probabilities
        confidence = loaded_model.predict_proba([features_list])

        # formulate the response to return to client
        result['prediction'] = int(prediction[0])
        result['confidence'] = str(round(np.amax(confidence[0]) * 100 ,2))
        
    return templates.TemplateResponse("index.html", {"request": request, "result": result})


