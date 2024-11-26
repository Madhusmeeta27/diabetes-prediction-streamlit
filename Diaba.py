import numpy as np
import pickle
import streamlit as st

# Load the trained model
loaded_model = pickle.load(open(r"C:\Users\madhu\Downloads\trained_model.sav", 'rb'))

# Define the prediction function
def diabetes_prediction(Pregnancies, Glucose, BloodPressure, Skin_Thickness, Insulin, BMI, DiabetesPedigeeFunction, Age):
    # Create a tuple of input data
    input_data = (Pregnancies, Glucose, BloodPressure, Skin_Thickness, Insulin, BMI, DiabetesPedigeeFunction, Age)
    # Convert input data to a numpy array
    input_data_as_nparray = np.asarray(input_data, dtype=float)
    # Reshape the array for prediction
    input_data_reshape = input_data_as_nparray.reshape(1, -1)
    # Make the prediction
    prediction = loaded_model.predict(input_data_reshape)
    # Return the result
    if prediction[0] == 0:
        return "Non-diabetic"
    else:
        return "Diabetic"

# Define the Streamlit app
def main():
    # Set the title
    st.title("Diabetes Prediction App")

    # Collect user input
    Pregnancies = st.text_input("Number of Pregnancies", "0")
    Glucose = st.text_input("Glucose Level", "0")
    BloodPressure = st.text_input("Blood Pressure", "0")
    Skin_Thickness = st.text_input("Skin Thickness", "0")
    Insulin = st.text_input("Insulin Level", "0")
    BMI = st.text_input("BMI", "0")
    DiabetesPedigeeFunction = st.text_input("Diabetes Pedigree Function", "0")
    Age = st.text_input("Age", "0")

    # Initialize prediction output
    Diagnosis = ''

    # Prediction button
    if st.button('Diabetes Test Result'):
        try:
            # Convert inputs to float for prediction
            Diagnosis = diabetes_prediction(
                float(Pregnancies), float(Glucose), float(BloodPressure), float(Skin_Thickness),
                float(Insulin), float(BMI), float(DiabetesPedigeeFunction), float(Age)
            )
        except ValueError:
            Diagnosis = "Please enter valid numerical inputs."

    # Display the result
    st.success(Diagnosis)

# Run the app
if __name__ == '__main__':
    main()
