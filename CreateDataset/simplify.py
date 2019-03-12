import torch
from torchvision import transforms
from torch.utils.serialization import load_lua

from PIL import Image
import argparse

class Sketch_Maker:
	def __init__(self):
		parser = argparse.ArgumentParser(description='Sketch simplification demo.')
		parser.add_argument('--model', type=str, default='model_gan.t7', help='Model to use.')
		opt = parser.parse_args()

		self.use_cuda = torch.cuda.device_count() > 0
		torch.cuda.set_device(0)
		cache  = load_lua(opt.model)
		self.model  = cache.model
		self.immean = cache.mean
		self.imstd  = cache.std
		self.model.evaluate()

	def sketchify(self, image):
		# Creates sketch from an image
		data  = image
		w, h  = data.size[0], data.size[1]
		pw    = 8-(w%8) if w%8!=0 else 0
		ph    = 8-(h%8) if h%8!=0 else 0
		data  = ((transforms.ToTensor()(data)-self.immean)/self.imstd).unsqueeze(0)
		if pw!=0 or ph!=0:
		   data = torch.nn.ReplicationPad2d( (0,pw,0,ph) )( data ).data

		if self.use_cuda:
		   pred = self.model.cuda().forward( data.cuda() ).float()
		else:
		   pred = self.model.forward( data )
		return pred[0]


