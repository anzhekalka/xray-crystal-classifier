"""
Streamlit web app for the X-Ray Crystal Classifier project.
Loads the trained ResNet-18 model, classifies uploaded diffraction 
pattern images into one of 7 crystal systems, and displays a Mistral 
AI scientific explanation.
"""

import streamlit as st
from torchvision import transforms
from PIL import Image
import torchvision.models as models
import torch.nn as nn
import torch 
import joblib
import plotly.graph_objects as go
from explainer import explain_prediction


@st.cache_resource
def load_model(): 
    "returns the trained model and classes"
    resnet18 = models.resnet18(weights = None)
    new_conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False) 
    resnet18.conv1 = new_conv1 # swaps the first layer to natively accept 1-channel black-and-white image

    in_features = resnet18.fc.in_features
    resnet18.fc = nn.Sequential(
        nn.Dropout(0.5),
        nn.Linear(in_features, 7)
    )

    resnet18.load_state_dict(torch.load("model/model.pth", map_location="cpu")) #loading the traind model weighs
    resnet18.eval()

    class_names = joblib.load('model/classes.pkl')
    return resnet18, class_names

model, class_names = load_model()

#UI
st.set_page_config(page_title="Crystal System Detector")
st.title("X-Ray Diffraction Crystal Classifier")
st.write("""
Upload an X-ray diffraction pattern image and this model will classify 
which of the 7 crystal systems produced it, using a ResNet-18 CNN 
fine-tuned on synthetic patterns generated from real Materials Project 
crystal structures.
""")
st.divider()

#image handling 
uploaded_file = st.file_uploader("Upload a diffraction pattern image", type=["png", "jpg", "jpeg"])
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.Grayscale(num_output_channels=1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.449], std=[0.226])
])
if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded pattern", width=300)

#button
if uploaded_file and st.button("Classify"): 
     img_tensor = transform(image).unsqueeze(0) #batch 
     with torch.no_grad():
        outputs = model(img_tensor)
        probabilities = torch.softmax(outputs, dim=1)[0] #a flat 1D array of 7 probabilities
        prediction = torch.argmax(probabilities).item() #index of the highest probability value in the array
        confidence = probabilities[prediction].item() #decimal percentage of that winning index 
        predicted_class = class_names[prediction] #Maps the winning index integer back to a text string

        #UI 
        st.success(f"Predicted crystal system: **{predicted_class.capitalize()}**")
        st.metric(label="Model Confidence", value=f"{confidence:.1%}")
        st.progress(confidence)
        fig = go.Figure(go.Bar (
            x = class_names, 
            y = probabilities.numpy(), 
            marker_color='#3a7bd5'
        ))
        fig.update_layout(title="Probability across all crystal systems")
        st.plotly_chart(fig, width='stretch') #automatic stretching on the screen 

        with st.spinner("Consulting the scinetists..."):
            explanation = explain_prediction(predicted_class, confidence)
            if explanation : 
                st.subheader("Scientific Explanation")
                st.markdown(explanation)
            else: 
                st.warning("AI explanation unavailable - check your API key")






