import streamlit as st
from transformers import TrOCRProcessor, VisionEncoderDecoderModel
from PIL import Image

# Define a function to perform OCR on the uploaded image
@st.cache_data(allow_output_mutation=True)
def perform_ocr(image):
    processor = TrOCRProcessor.from_pretrained('microsoft/trocr-base-handwritten')
    model = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-base-handwritten')
    
    # Process the uploaded image
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    
    # Generate text from the processed image
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return generated_text

# Streamlit app
st.title("Handwritten Text Recognition")

# Upload image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Display the uploaded image
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

    # Perform OCR on the uploaded image
    generated_text = perform_ocr(Image.open(uploaded_image).convert("RGB"))

    # Display the generated text
    st.subheader("Generated Text:")
    st.write(generated_text)
