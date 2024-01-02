from prepro import Image_reco
import torch


image_object = Image_reco("train", ["cat", "dog","wild"])

print(torch.backends.mps.is_available())

dataframe = image_object.getting_data()
train_gen,val_gen, test_gen = image_object.preprocessing_data(dataframe)
model = image_object.build_model()
x = image_object.train_model(model,train_gen,val_gen,test_gen)

