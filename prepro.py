import torch # Main PyTorch Library
from torch import nn # Used for creating the layers and loss function
from torch.optim import Adam # Adam Optimizer
import torchvision.transforms as transforms # Transform function used to modify and preprocess all the images
from torch.utils.data import Dataset, DataLoader # Dataset class and DataLoader for creating the objects
from sklearn.preprocessing import LabelEncoder # Label Encoder to encode the classes from strings to numbers
import matplotlib.pyplot as plt # Used for visualizing the images and plotting the training progress
from PIL import Image # Used to read the images from the directory
import pandas as pd # Used to read/create dataframes (csv) and process tabular data
import numpy as np # preprocessing and numerical/mathematical operations
import os # Used to read the images path from the directory
LR = 1e-5
BATCH_SIZE = 20
EPOCHS = 10

label_encoder=LabelEncoder() #change the classes "dog,cat,wild"into"1,2,3"
device = "mps"#for MAC users

#device = "cuda" if torch.cuda.is_available() else "cpu" 
#for Windows users detect the GPU if any, if not use CPU, change cuda to mps if you have a mac
total_loss_train_plot = []
total_loss_validation_plot = []
total_acc_train_plot = []
total_acc_validation_plot = []
class Image_reco:
    
    def __init__(self, data_path, labels):
        self.data_path=data_path
        self.labels=labels
    def getting_data(self):
        Data = []
        classes = []
        file_names = os.listdir(self.data_path)

        for i in (file_names):
            for j in range(len(self.labels)):
                if self.labels[j] in i:
                    classes.append(self.labels[j])
                else:
                    continue
        for i in range(len(file_names)):
            full_path = os.path.join(self.data_path, file_names[(i)])
            Data.append(full_path)

        final_data = pd.DataFrame(list(zip(Data, classes)), columns = ["image_path", "labels"])
        self.final_data=final_data

        final_data['labels'] = label_encoder.fit_transform(final_data['labels']) # here is it

        np.save('label_encoder.npy', label_encoder.classes_)
        return final_data
    
    def preprocessing_data(self,dataframe):
        train=dataframe.sample(frac=0.7,random_state=7) # Create training of 70% of the data
        test=dataframe.drop(train.index) # Create testing by removing the 70% of the train data which will result in 30%

        val=test.sample(frac=0.5,random_state=7) # Create validation of 50% of the testing data
        test=test.drop(val.index) # Create testing by removing the 50% of the validation data which will result in 50%
        
        
        transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.ConvertImageDtype(torch.float) 
        ])

        train_dataset = CustomImageDataset(dataframe=train, transform=transform)
        val_dataset = CustomImageDataset(dataframe=val, transform=transform)
        test_dataset = CustomImageDataset(dataframe=test, transform=transform)
        return (train_dataset,val_dataset,test_dataset)
    def build_model(self):
        model = Net(len(self.final_data['labels'].unique())).to(device)
        # Define loss function and optimizer
        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = Adam(model.parameters(), lr=LR)
        return model
    
    def train_model(self,model,train_dataset,val_dataset,test_dataset):
        train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
        model=self.build_model() #NICEEEE
        for epoch in range(EPOCHS):
            total_acc_train = 0
            total_loss_train = 0
            total_loss_val = 0
            total_acc_val = 0
            for (inputs, labels) in train_loader:

                self.optimizer.zero_grad()
                outputs = model(inputs)
                train_loss = self.criterion(outputs, labels)
                total_loss_train += train_loss.item()
                train_loss.backward()
                train_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
                total_acc_train += train_acc
                self.optimizer.step()
            with torch.no_grad(): #use the model without changing the weights
             for i, (inputs, labels) in enumerate(val_loader):
              outputs = model(inputs)
              val_loss = self.criterion(outputs, labels)
              total_loss_val += val_loss.item()

              val_acc = (torch.argmax(outputs, axis = 1) == labels).sum().item()
              total_acc_val += val_acc
            print(f'Epoch {epoch+1}/{EPOCHS}, Train Loss: {round(total_loss_train/100, 4)} Train Accuracy {round((total_acc_train)/train_dataset.__len__() * 100, 4)} Validation Loss: {round(total_loss_val/100, 4)} Validation Accuracy: {round((total_acc_val)/val_dataset.__len__() * 100, 4)}')      
        
        torch.save(model, "final_model.pth")
    
        return model
class CustomImageDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.dataframe = dataframe
        self.transform = transform
        self.labels = torch.tensor(list(dataframe['labels']),dtype=torch.long).to(device) 


    def __len__(self):
        return self.dataframe.shape[0]

    def __getitem__(self, idx):
        img_path = self.dataframe.iloc[idx, 0]
        label = self.labels[idx]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image).to(device)

        return image, label 

class Net(nn.Module):
    def __init__(self,outputs):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 32, kernel_size = 3, stride = 1, padding = 1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size = 3, stride = 1, padding = 1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size = 3, stride = 1, padding = 1)
        
        self.pooling = nn.MaxPool2d(2,2)
        self.relu = nn.ReLU()

        self.flatten = nn.Flatten()
        self.linear = nn.Linear(32768, 128)
        self.output = nn.Linear(128, outputs)


    def forward(self, x):
        x = self.conv1(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.conv3(x)
        x = self.pooling(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear(x)
        x = self.output(x)

        return x

        
    
    
