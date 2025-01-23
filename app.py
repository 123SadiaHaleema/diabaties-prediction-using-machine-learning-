from flask import Flask, render_template, request, jsonify
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Load the diabetes dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv"
column_names = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age', 'Outcome']
data = pd.read_csv(url, header=None, names=column_names)

# Separate features and target variable
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Create the model (Decision Tree Classifier)
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get user input from form
    pregnancies = int(request.form['Pregnancies'])
    glucose = int(request.form['Glucose'])
    blood_pressure = int(request.form['BloodPressure'])
    skin_thickness = int(request.form['SkinThickness'])
    insulin = int(request.form['Insulin'])
    bmi = float(request.form['BMI'])
    diabetes_pedigree_function = float(request.form['DiabetesPedigreeFunction'])
    age = int(request.form['Age'])

    # Make prediction
    input_data = scaler.transform([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, diabetes_pedigree_function, age]])
    prediction = model.predict(input_data)[0]

    # Interpret prediction
    if prediction == 1:
        prediction_text = 'High likelihood of diabetes.'
    else:
        prediction_text = 'Low likelihood of diabetes.'

    return render_template('result.html', prediction_text=prediction_text)

if __name__ == '__main__':
    app.run(debug=True)
