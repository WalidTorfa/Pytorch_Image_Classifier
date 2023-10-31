from prepro import Image_reco
import torch
device = "cuda" if torch.cuda.is_available() else "cpu" 
print(device)

image_object = Image_reco("train", ["cat", "dog","wild"])

dataframe = image_object.getting_data()
train_gen,val_gen, test_gen = image_object.preprocessing_data(dataframe)
model = image_object.build_model()
x = image_object.train_model(model,train_gen,val_gen,test_gen)