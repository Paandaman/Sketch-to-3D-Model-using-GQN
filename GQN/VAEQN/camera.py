import torch
import numpy as np

class Camera_view:
    # Object which keeps track of the camera view and generates new ones
    def __init__(self, start_view):
        start = torch.tensor([start_view[0], start_view[1], start_view[2], start_view[3], start_view[5], start_view[4], start_view[6], start_view[7]]).cuda()
        self.view = start
        self.phi = np.arccos(start_view[3])
        self.theta = np.arccos(start_view[5])

    def get_view(self):
        # Change view point by rotating 0.05 radians in both polar and azimuthal direction
        r = 2 # radius
        self.phi = (self.phi + 0.05)
        self.theta = (self.theta + 0.05)%(np.pi)
        self.view[0] = r*np.sin(self.theta)*np.cos(self.phi) # x
        self.view[1] = r*np.sin(self.theta)*np.sin(self.phi) # y
        self.view[2] = r*np.cos(self.theta) # z
        self.view[3] = np.cos(self.phi) 
        self.view[4] = np.sin(self.phi)
        self.view[5] = np.cos(self.theta)
        self.view[6] = np.sin(self.theta)
        return self.view

    def plot_grid(self, theta, phi):
        # Change view point by specified angles theta and phi
        r = 2 # radius
        self.view[0] = r*np.sin(theta)*np.cos(phi)
        self.view[1] = r*np.sin(theta)*np.sin(phi)
        self.view[2] = r*np.cos(theta)
        self.view[3] = np.cos(phi) 
        self.view[4] = np.sin(phi)
        self.view[5] = np.cos(theta)
        self.view[6] = np.sin(theta)
        return self.view