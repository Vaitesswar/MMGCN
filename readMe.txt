%%% Generating input data
The  json skeleton data for NW+UCLA dataset was obtained from the following link and was provided in CTR-GCN's Github page.
https://github.com/Uason-Chen/CTR-GCN

The skeleton data for NTU RGB+D and NTU RGB+D 120 datasets was obtained by extracting the 3D coordinates from raw .skeleton files obtained
from NTU ROSE lab.
https://rose1.ntu.edu.sg/dataset/actionRecognition/

%%% Models
The GCN models can be found under "models" folder for each dataset.

 
%%% Model training
The model architecture has to be modified in line 25 of training.py file in training folder. The parameters of the optimizer, the training
schedule and path to input data can be adjusted in main.py file. "processor" class will take in the parameters and will start training the 
model upon initiating "processor.start" command.
