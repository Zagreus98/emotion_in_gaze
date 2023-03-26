# emotion_in_gaze
Training code for the emotion_in_gaze model.

TODO:
  - Create a common data format for gaze and emotion datasets
  - Create a dataloader which takes into consideration from which dataset the image is
  - Define transforms
  - Train loop/validation loop
  - Implement total loss for gaze and emotion classification
    - If data from xgaze: semi-supervized loss for classification + Gaze_loss
    - If data from rafdb: semi-supervized loss for regression (will be 0 for the moment, research in progress) + Classification loss
    
