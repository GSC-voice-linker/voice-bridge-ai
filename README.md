This project can be used to generate the necessary data for training a model to recognize sign language, train the model, and then test the trained model's ability to recognize sign language in real-time using a webcam. 
Some parts of the TestAllModel.py within this project have been modified and are currently uploaded to the server.

1. MakeAllDataset.py
MakeAllDataset.py is the code responsible for generating each sign language dataset. It selects the classes of sign language data to train, selects the number of frames for each data, and generates the data variably. 
It typically requires Google's open platform, such as Mediapipe, as well as NumPy, OpenCV, and other dependencies.

2. TrainAllModel.py
TrainAllModel.py is the code used to build and train an LSTM model using the sign language data generated earlier. 
It requires sklearn and TensorFlow for constructing and training the LSTM model.

3. TestAllModel.py
TestAllModel.py is the code used to utilize the trained model to recognize sign language from the user in real-time using a webcam and accurately convert it into corresponding text.