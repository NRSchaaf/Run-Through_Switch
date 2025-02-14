import streamlit as st
from PIL import Image
import torch
from torchvision import transforms
import torch.nn as nn

st.title("Train Switch Image Classifier")
st.subheader("Which path will the train take?")
st.image("rs00007.jpg", caption="Example 1")
st.text("""Trains in rail yards travel on average 5-7 MPH, however regardless of the speed accidents still occur frequently. To address these potentially costly and time-consuming issues, our model is designed to analyze a picture of an upcoming track and identify whether the current path contains a closed switch, open switch, or straight path.""")

st.write("Upload an image of train tracks, and our model will classify it!")

#for user uploads
uploaded_file = st.file_uploader("Upload your image here", type=["jpg", "jpeg", "png"])

# Prediction Section
if uploaded_file:
    image = Image.open(uploaded_file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)
    st.write("Analyzing the image...")

    #Optimal model weights from previous training 
    class CNNModel(nn.Module):
        def __init__(self):
            super(CNNModel, self).__init__()
            self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
            self.fc1 = nn.Linear(64 * 16 * 16, 128)
            self.fc2 = nn.Linear(128, 3)  # 3 classes: open_switch, closed_switch, straight

        def forward(self, x):
            x = nn.ReLU()(self.conv1(x))
            x = nn.MaxPool2d(2)(x)
            x = nn.ReLU()(self.conv2(x))
            x = nn.MaxPool2d(2)(x)
            x = nn.ReLU()(self.conv3(x))
            x = nn.MaxPool2d(2)(x)
            x = x.view(x.size(0), -1)  # Flatten
            x = nn.ReLU()(self.fc1(x))
            x = self.fc2(x)
            return x
  
    model = CNNModel()
    try:
        model.load_state_dict(torch.load('railroad_track_cnn.pth'))
        model.eval() 
    except Exception as e:
        st.error(f"Error loading model: {e}")
        st.stop()

    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image_tensor = transform(image).unsqueeze(0)  

    try:
        with torch.no_grad():
            output = model(image_tensor)
            _, predicted = torch.max(output, 1) 

        # Predicition mapping
        label_map = {0: 'open_switch', 1: 'closed_switch', 2: 'straight'}
        predicted_class = label_map[predicted.item()]

        st.write(f"**Prediction:** {predicted_class}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")

else:
    st.write("Please upload an image to get started.")
