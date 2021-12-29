from fastapi import FastAPI, Form
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import pickle
import sklearn


app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
#Как передавать 6 аргументов , а не 4

# , Product_Category_2 = Form(...), Product_Category_3 = Form(...)
@app.post("/request/")
def request(Age= Form(...),Gender=Form(...),Occupation = Form(...), Marital_Status = Form(...), Stay_In_Current_City_Years = Form(...),City_Category=Form(...)):
#  ,"Product_Category_2": [1] ,"Product_Category_3": [2]
    dataframe = pd.DataFrame({"Age":Age,"Gender":[Gender],"Occupation": [Occupation], "Marital_Status": [Marital_Status], "Stay_In_Current_City_Years": [Stay_In_Current_City_Years],"City_Category":[City_Category]})
    model_ridge = pickle.load(open("model.sav", 'rb'))
    result = model_ridge.predict(dataframe).round(2)


    return {"result": result[0]}
