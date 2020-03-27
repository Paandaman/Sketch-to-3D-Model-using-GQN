# Sketch-to-3D-Model-using-GQN
The code used to produce the results in my thesis found at ![alt text](urn:nbn:se:kth:diva-251507) .


# Create Dataset - Turn 3D models into 2D sketches
Requirements: Blender and pre-trained GAN model named "model_gan.t7" from https://esslab.jp/~ess/en/code/sketch_code/ . Sketch clean up network is based on: https://github.com/bobbens/sketch_simplification by Edgar Simo-Serra, Satoshi Iizuka, Kazuma Sasaki and Hiroshi Ishikawa.

Steps:
1: Capture images of 3D models by rendering them in Blender from different view points. Specify the paths in capture_images.py and then run it as follows:
blender test.blend -P capture_images.py
The images are captured from 15 angles and saved together in a folder to represent a "scene".

2: Turn the images into sketches. Make sure you have the "model_gan.t7" in your directory, then specify the paths in sketchify.py and run the script.

3: Convert all files into PyTorch tensor format and group sketches and ground truths(the images obtained before running sketchify) together. Run convertShapeNet2torch.py with the necessary arguments specified.

4. Unzip all files, e.g. by running "gunzip *".

Example model from the ShapeNet dataset:

![alt text](https://github.com/Paandaman/Sketch-to-3D-Model-using-GQN/blob/master/sketch.png)

# GQN model
The GQN implementation using the Convolutional DRAW is based on the work done by https://github.com/iShohei220 , with the modifications needed to render images from sketches. Training the model is computationally expensive and producing the results seen in the report required 16 days and 23 hours of training on a Nvidia GTX 1080 Ti. I got the advice from github users iShohei220 and brettgohre that setting the rollout length of the LSTM (L) to 8 instead of 12, lowering the learning rate by a magnitude of 10 and using shared cores in the LSTM helped speeding up the learning process. This does however also reduce the quality of the final results so for the results seen in the paper, the rollout length was set to 12. 

The training script can e.g. be executed as follows:

python3 train.py --train_data_dir /path/to/SketchDataset/Train --test_data_dir /path/to/SketchDataset/Test --log_dir /path/to/log/dir --shared_core True

Example of a single, real sketch rendered from various view points:

![alt text](https://github.com/Paandaman/Sketch-to-3D-Model-using-GQN/blob/master/Singlesketch.png)

Example of images generated in real time below. Links to a YouTube video:

[![alt text](https://img.youtube.com/vi/WChV4mcz8dc/0.jpg)](https://www.youtube.com/watch?v=WChV4mcz8dc "Sketch to 3D Model")
