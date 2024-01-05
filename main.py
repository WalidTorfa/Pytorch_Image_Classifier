from prepro import Image_reco
import torch


# Ask the user to input a series of labels, separated by commas
user_input = input("Enter labels separated by commas: ")

# Split the string into a list at each comma
labels_list = user_input.split(',')

# Optionally, strip whitespace from each label
labels_list = [label.strip() for label in labels_list]






image_object = Image_reco("train", labels_list)

print(torch.backends.mps.is_available())

dataframe = image_object.getting_data()
train_gen,val_gen, test_gen = image_object.preprocessing_data(dataframe)
model = image_object.build_model()
x = image_object.train_model(model,train_gen,val_gen,test_gen)

