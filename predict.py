import torchvision.transforms as transforms # Transform function used to modify and preprocess all the images
from PIL import Image # Used to read the images from the directory
import torch
from sklearn.preprocessing import LabelEncoder
import numpy as np

le = LabelEncoder()
le.classes_ = np.load('label_encoder.npy', allow_pickle= True)

img_path = input("input the image path:")
image = Image.open(img_path).convert('RGB')

transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float) 
        ])

image = transform(image)

# print(image.shape)

image = image.unsqueeze(0) # Set the batch size to 1 so final dimention is (1,3,128,128)

# print(image.shape)

model = torch.load('my_entire_model.pth')


output = model(image)
numerical_output = (torch.argmax(output, axis = 1).item())
final_output = le.inverse_transform([numerical_output])
print(final_output)