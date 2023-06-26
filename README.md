# 1. Neon dance:
## Requirements:
  ```pip install -r neon_dance_requirements.txt```
## Instruction (Read file video_neon_dance.py for using more parameters)
  ```python3 video_neon_dance.py -i "path/to/input/video" -o "path/to/output/video"```

# 2. VIDEO BLENDING BASED ON EYE STATE
## Requirements:
Please have opencv, numpy, matplotlib and mediapipe installed on your device
## Instructions:
*End to end:* ```python test.py``` (the result will be saved in result.gif file)<br>
*Segmentation:* ```python segmentation.py``` <br>
*Draw Brute force plot:* ```python plot_eye_state.py``` <br>
*Speed test:* ```python speed_capture.py```

# 3. CHANGE BACKGROUND.  
## Requirements:  
Please install opencv, numpy, cvzone, os, imageio before run code.
## Instructions:
* In change_background:  
The gif folder contains file.gif, you can add gif to that folder. Note the size of the gif file.  
```python change_background.py``` to run change background model

# 4. Goku effects
## Requirements:
Please have opencv, numpy, mediapipe and PIL already installed on your local workspace.
## Instructions:
*End to end:* ```main.py```. This's also the main source.
The image and gif inputs are placed in the same level with source ```main.py```, do not group them.
