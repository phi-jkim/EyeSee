# EyeSee
EyeSee is a project where we integrate classification using torchvision's SSDLiteMobileNet model for bounding boxes of snow which can be hazardous if unnoticed in the winter. The device performs inference on-device on a Raspberry PI. 

We use visible light and an infrared camera attached to a pair of glasses to capture real-time images from the cameras. 

These cameras are then fed into a lightweight inference pipeline and using the CPU of the raspberry PI, it will alert the user with a buzzing sound if it classifies and notices snow as a potential threat in the scene. 

## Project Structure 
To install the necessary dependencies 
```
pip3 install -r requirements.txt 
```
To train or test the ML scripts, first clone the repository and run the training or inference scripts 
```
cd EyeSee 
python inference_annotate.py 
python train_annotate.py     
```

## Try it out! 
Try training on our self-annotated snowing data with our training or inference script under the directory EyeSee! 

You can also connect a raspberry pi and try using our CameraTest.py and inference files to alert users with warnings. 

<!-- The project structure is detailed below 
    
    .
    ├── EyeSee                  # Inference pipeline for pretrained weights and data 
    ├── software_test           # Software tests on utils and image manipulations 
    ├── camera_inference        # Integration 
    ├── ...
    ├── requirements.txt        # Dependencies for installation 
    └── README.md -->

Inside EyeSee the ML pipeline 
    
    .
    ├── data                    # Dataset for annotations and classification
    │   ├── config              # Configuration file for yolo data format
    │   ├── *.jpg               # .jpg images 
    │   └── *.txt               # .txt annotations
    ├── pretrained_weights      # Pytorch weights from training 
    ├── results                 # Resulting images from inference 
    ├── utils                   # Tools for data loading and image processing 
    │   ├── struct              # Dataloader for machine learning training                
    │   ├── tools               # Utils for dataloading
    ├── inference_annotate      # Testing script 
    ├── train_annotate          # Training script
    └── ...




Install necessary dependencies on requirements.txt 