# Sketch-to-3D-Model-using-GQN
The code used to produce the results in the enclosed thesis.


# Create Dataset
Requirements: Blender and pre-trained GAN model named "model_gan.t7" from https://esslab.jp/~ess/en/code/sketch_code/ . Sketch clean up network is based on: https://github.com/bobbens/sketch_simplification by Edgar Simo-Serra, Satoshi Iizuka, Kazuma Sasaki and Hiroshi Ishikawa.

# Steps:
1: Capture images of 3D models by rendering them in Blender from different view points. Specify the paths in capture_images.py and then run it as follows:
blender test.blend -P capture_images.py
The images are captured from 15 angles and saved together in a folder to represent a "scene".

2: Turn the images into sketches. Make sure you have the "model_gan.t7" in your directory, then specify the paths in sketchify.py and run the script.

3: Convert all files into PyTorch tensor format and group sketches and ground truths(the images obtained before running sketchify) together. Run convertShapeNet2torch.py with the necessary arguments specified.
