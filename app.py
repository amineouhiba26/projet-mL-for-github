import streamlit as st
from PIL import Image
from predict_model import predict_accessory

def main():
    st.title("🔌 Accessory Classifier")
    st.markdown("Upload an image of an accessory (Charger, Headphone, etc.) to get a prediction.")

    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert('RGB')
        st.image(image, caption='Uploaded Image', use_column_width=True)
        st.write("Classifying...")

        try:
            prediction, confidence = predict_accessory(uploaded_file)
            st.success(f"Prediction: **{prediction}**")
            st.info(f"Confidence: **{confidence:.2%}**")
        except Exception as e:
            st.error(f"Error during prediction: {str(e)}")

if __name__ == "__main__":
    main()
