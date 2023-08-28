import streamlit as st
import pickle
from preprocess_text import preprocess_text  # Import preprocessing function


# Load the serialized model
model_filename = 'mbti500_SVCmodel.sav'
with open(model_filename, 'rb') as model_file:
    text_clf = pickle.load(model_file)


# Create the Streamlit app
def main():
    st.title("MBTI Personality Type Predictor")
    st.write("Enter your text and we'll predict your personality type!")

    # User input
    user_input = st.text_area("Enter your text here:")

    if st.button("Predict"):
        if user_input:
            # Preprocess user input
            preprocessed_input = preprocess_text(user_input)

            # Make a prediction using the loaded model
            prediction = text_clf.predict([preprocessed_input])[0]
            st.write(f"Predicted Personality Type: {prediction}")


if __name__ == "__main__":
    main()
