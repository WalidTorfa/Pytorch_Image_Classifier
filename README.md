Usage Instructions

Running the Main Script
To utilize the main functionality of this repository, simply execute main.py. 
Upon running, the script will prompt you for the labels you wish to use for classifying your dataset. 
Please enter these labels separated by commas. Note: Do not enclose the labels in quotation marks.
main.py initiates a training process for 10 epochs by default. 
This setting is suitable for a variety of applications but can be modified if your project requires more extensive training.
To adjust the number of epochs, please modify the corresponding setting in the prepro.py file.
After the training process is completed, the model will be automatically saved as a .pth file.
This facilitates future usage of the model for predictions, eliminating the need for retraining each time.

Using the Prediction Script
For predictions using an already trained model, the predict.py script is your go-to tool.
To use this, you will need to provide the path of the image you want to classify.
The script will then employ the previously saved model to perform the classification on the specified image.

the "final_model.pth" in the repo was trained to indetify wether the given picture of an animal face is a "Cat" or "Dog" or "Wild"



