from flask import Flask,redirect,url_for,render_template,request
import pickle
from datetime import datetime,timezone
from flask_cors import CORS

app=Flask(__name__)

CORS(app,resources={r"/*":{"origins":"*"}})

with open("model.pkl", "rb") as f:
    model = pickle.load(f)

@app.route('/add',methods=["POST"])
def welcome():
   utc_now = datetime.now(timezone.utc)
   dob_string= request.json["Date of Birth"]
   dob = datetime.fromisoformat(dob_string[:-1])  
   dob = dob.replace(tzinfo=timezone.utc)
   difference = utc_now - dob
   difference_in_days = difference.days
   gender=request.json["Gender"]
   height=request.json["Height"]
   weight=request.json["Weight"]
   active=request.json["active"]
   smoke=request.json["Smoke"]
   systolic_bp=request.json["Systolic_BP"]
   diastolic_bp=request.json["Diastolic_BP"]
   cholestrol=request.json["Cholestrol"]
   glucose=request.json["Glucose"]
   alcohol=request.json["Alcohol"]
   if gender=="Male":
      g=2
   else:
      g=1
   if active=="Yes":
      act=1
   else:
      act=0
   if smoke=="Yes":
      sm=1
   else:
      sm=0
   if alcohol=="Yes":
      al=1
   else:
      al=0
   s=""
   if(active=="Yes"):
       s=s+"You are physically Active"
   else:
      s=s+"Try to be more physically Active"
   if(smoke=="Yes"):
       s=s+" You smoke regularly Quit Smoking"
   else:
      s=s+" You don't smoke"
   if(systolic_bp>=140):
       s=s+" You have high systolic blood pressure Reduce your salt (sodium) intake and follow a heart-healthy eating pattern "
   elif(systolic_bp>130 and systolic_bp<140):
      s=s+" Caution!!You are on border line Reduce your salt (sodium) intake and follow a heart-healthy eating pattern "
   else:
      s=s+" You have great systolic blood pressure"
   if(diastolic_bp>=90):
       s=s+" You have high diastolic blood pressure Avoid Sodium, Sugar, and Processed Foods and try Low-Impact Exercise"
   elif(diastolic_bp>80 and systolic_bp<90):
      s=s+" Caution!!You are on border line Avoid Sodium, Sugar, and Processed Foods and try Low-Impact Exercise"
   else:
      s=s+" You have great diastolic blood pressure"  
   if(cholestrol<200):
       s=s+" You have normal cholesterol"
       cholestrol=1
   elif(cholestrol>="200" and cholestrol<"240"):
      s=s+" Caution!!You are on border line Eat soluble fiber, monosaturated fats and limit trans fat"
      cholestrol=2
   else:
      s=s+" You have high cholesterol  Eat soluble fiber, monosaturated fats and limit trans fat"
      cholestrol=3
   if(glucose<100):
       s=s+" You have normal glucose levels"
       glucose=1
   elif(glucose>=100 and glucose<125):
      s=s+" Caution!!You are on border line Eat foods that are rich in chromium and magnesium having more fiber and drink plenty of water"
      glucose=2
   else:
      s=s+" You have high glucose levels  Eat foods that are rich in chromium and magnesium having more fiber and drink plenty of water"
      glucose=3
   data_list=[[difference_in_days,g,height,weight, systolic_bp,diastolic_bp,cholestrol,glucose,sm,al,act]]
   prediction=model.predict(data_list)
   if prediction[0]==0:
      s=s+" Congratulations  You have a healthy heart  Keep Going!!"
   else:
       s=s+ "You are prone to Heart diseases . Change your Lifestyle"
   return {"Speech":s}
   
   
 
if __name__ =='__main__':
  app.run(debug=True)