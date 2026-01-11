import pickle
from flask import Flask, request, jsonify, app, url_for, render_template 
import pandas as pd 
import numpy as np 

app = Flask(__name__)   # its just a starting point 
## Load the model 
regmodel = pickle.load(open("regmodel.pkl", "rb"))
Scaler = pickle.load(open("Scaling.pkl", "rb"))

@app.route('/')
def home():
    return render_template('home.html')

# we created a Scaling.pkl because we want to first standardise the entire data then we will do the prediction
@app.route('/predict_api', methods=['POST'])          # This creates an endpoint called /predict   # So we have created a predcit api 
def predict_api():
    data = request.json['data']            # This reads the data you send from Postman in JSON format.
    print(data)
    print(np.array(list(data.values())).reshape(1,-1))   # Getting dictionary values by converting data into list and reshaped it by converting into numpy array 
    new_data = Scaler.transform(np.array(list(data.values())).reshape(1,-1))
    output = regmodel.predict(new_data)
    print(output[0])   # As it would be in two dimensional array so take the first value
    return jsonify(output[0])

# Now create a web application with a form where we will add values and predict the results 
@app.route('/predict', methods=['POST'])
def predict():
    data = [float(x) for x in request.form.values()]
    final_input=Scaler.transform(np.array(data).reshape(1,-1))
    print(final_input)
    output = regmodel.predict(final_input)[0]
    return render_template("home.html", prediction_text="The House Price prediction is {}".format(output))


if __name__=="__main__":
    app.run(debug=True)