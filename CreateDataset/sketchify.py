import glob
import numpy as np
from PIL import Image
from PIL import ImageFilter
import PIL.ImageOps
from simplify import *
import shutil
import os
import pathlib
import shutil

def main():
	dir =  "path/to/preprocessed/data/"
	output_dir = "/path/to/output_dir"
	model_nr = 0
	samples_per_model = 15
	sketch_it = Sketch_Maker()
	transform = transforms.Compose([transforms.ToPILImage(),transforms.Resize(size=128),transforms.ToTensor()])
	for name in glob.glob(dir+"*/"): 
		model_nr_id = name.split("/")[-2]
		model_nr += 1
		path = output_dir + model_nr_id + "/"
		try:
			os.mkdir(path)
		except FileExistsError:
			pass
		for i in range(samples_per_model):
			model_nr_id_view = model_nr_id + "/" + model_nr_id +"_" + str(i) + ".jpg"

			try:
				image = Image.open(dir + model_nr_id_view)
			except FileNotFoundError:
				shutil.rmtree(path, ignore_errors=True)
				break

			imageWithEdges = image.filter(ImageFilter.FIND_EDGES)
			invertedEdges = PIL.ImageOps.invert(imageWithEdges)
			sketch = sketch_it.sketchify(invertedEdges)
			
			out_path = output_dir + model_nr_id + "/" + model_nr_id +"_" + str(i) + ".jpg"
			
			save_image( sketch, out_path )

			#save camera view , just moves the files to the new dir
			save_name = model_nr_id + "/" + model_nr_id +"_" + str(i) + ".txt"
			txt_name = dir + save_name
			out_path_txt = output_dir + save_name 
			shutil.copyfile(txt_name, out_path_txt)

main() 