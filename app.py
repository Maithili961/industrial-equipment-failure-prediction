
from flask import Flask, render_template, request
import joblib
import pandas as pd

app = Flask(__name__)

model = joblib.load("trained_model.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    result = None

    if request.method == "POST":
        data = {
            "Type": request.form["type"],
            "Air temperature [K]": float(request.form["air_temp"]),
            "Process temperature [K]": float(request.form["process_temp"]),
            "Rotational speed [rpm]": float(request.form["rpm"]),
            "Torque [Nm]": float(request.form["torque"]),
            "Tool wear [min]": float(request.form["tool_wear"])
        }

        input_data = pd.DataFrame([data])
        prediction = model.predict(input_data)[0]

        if prediction == 1:
            result = "FAILURE PREDICTED"
        else:
            result = "NO FAILURE PREDICTED"

    return render_template("index.html", result=result)

if __name__ == "__main__":
    app.run(debug=True)
