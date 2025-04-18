import streamlit as st
import pickle
from PIL import Image

# Load your model, vectorizer, and label encoder
with open("cuisine_model.pkl", "rb") as f:
    model = pickle.load(f)
with open("vectorizer.pkl", "rb") as f:
    vectorizer = pickle.load(f)
with open("label_encoder.pkl", "rb") as f:
    label_encoder = pickle.load(f)

# Set a custom background color
page_bg = """
<style>
body {
    background-color: #f3f4f6; /* Soft gray background */
    color: #333333; /* Text color */
    font-family: 'Arial', sans-serif; /* Clean font style */
}
input[type="text"] {
    border-radius: 10px; /* Rounded input fields */
    border: 1px solid #dddddd; /* Light border for input */
    padding: 10px; /* Add padding */
}
textarea {
    border-radius: 10px;
    border: 1px solid #dddddd;
    padding: 10px;
}
button {
    border-radius: 8px;
    background-color: #2196F3; /* Fancy blue button */
    color: white;
    border: none;
    padding: 8px 15px;
    font-size: 16px;
    cursor: pointer;
}
</style>
"""
st.markdown(page_bg, unsafe_allow_html=True)

# App title and description
st.title("🍴 Cuisine Predictor")
st.markdown("""
Welcome to the **Cuisine Predictor**!  
This app uses **Machine Learning** to identify the most likely cuisine based on the ingredients you provide.  
Just enter at least three ingredients, and we'll reveal the cuisine magic! 🎉  
""")

# Add a placeholder image
st.image(Image.open("cuisine_banner.jpeg"), use_column_width=True, caption="Discover World Cuisines with AI 🍲")

# Input section
st.markdown("### Enter Your Ingredients:")
ingredients = st.text_input("Type your ingredients (comma-separated, e.g., salt, garlic, soy sauce):")

# User interaction for prediction
if st.button("🎯 Predict Cuisine"):
    if len(ingredients.split(",")) >= 3:  # Ensure at least three ingredients
        # Preprocess user input
        processed_ingredients = [" ".join(ingredients.split(","))]  # Join as a single string
        vectorized_ingredients = vectorizer.transform(processed_ingredients)  # Vectorize the input
        
        # Predict the cuisine
        prediction_numeric = model.predict(vectorized_ingredients)[0]
        predicted_cuisine = label_encoder.inverse_transform([prediction_numeric])[0]

        # Display results
        st.success(f"✨ Based on the ingredients, the predicted cuisine is: **{predicted_cuisine}** 🍽️")
    else:
        st.error("⚠️ Please enter **at least 3 ingredients** to make a prediction!")

# Add a footer for flair
st.markdown("---")
st.markdown("""
### About  
Unleash your creativity and discover what cuisines match your favorite ingredients. Bon appétit! 🥂  
Developed with ❤️ using **Streamlit**.
""")
