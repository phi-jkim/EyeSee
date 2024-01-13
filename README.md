# EyeSee
EyeSee is a project where we integrate classification (multi-grid bounding box with torchvision's SSD) of snow which can be hazardous if unnoticed in the winter. The device performs inference on-device on a Raspberry PI. 

We use visible light and an infrared camera attached to a pair of glasses to capture real-time images from the cameras. 

These cameras are then fed into a lightweight inference pipeline with additional optimizations such as gamma correction (balancing low-light scenes with brigther images) and using the CPU of the raspberry PI, it will alert the user with a LED if it classifies snow in the scene. 

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

The project structure is detailed below 
    
    .
    ├── EyeSee                  # Inference pipeline for pretrained weights and data 
    ├── software_test           # Software tests on utils and image manipulations 
    ├── camera_inference        # Integration 
    ├── ...
    ├── requirements.txt        # Dependencies for installation 
    └── README.md

Inside EyeSee the ML pipeline 
    
    .
    ├── ...
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