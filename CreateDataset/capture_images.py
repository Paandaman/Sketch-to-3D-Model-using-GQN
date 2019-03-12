import bpy
import numpy as np
import mathutils
import math
import glob
import os


# Run script : blender test.blend -P capture_images.py

DATASET_PATH = "/path/to/ShapeNetCore"# Path to 3D models
OUTPUT_DIR = "/save_models/"#"." # Path to output directory
FILE_NAME =  "model_normalized.obj" # Name of 3D model files
LIGHTS_COUNT = 1 # Number of lights in scene (default: 1)
RES_X = 512 # Resolution in width
RES_Y = 512 # Resolution in height
BG_PATH = "/White_full1.jpg"#"." # Path to background image
TX_PATH = "/White_full1.jpg"#"." # Path to texture image

def load_obj(file_path):
	#load object
	full_path_to_file = file_path
	bpy.ops.import_scene.obj(filepath=full_path_to_file)
	#return handle to that object
	return bpy.context.selected_objects[0]

def load_img(filepath):
	#load image 
	return bpy.data.images.load(filepath)

def random_background():
	#load image
	img = load_img(BG_PATH)
	bpy.context.scene.use_nodes = True
	for n in bpy.context.scene.node_tree.nodes:
	    #find background image node
	    if n.type == 'IMAGE':
	        n.image = img

def add_material(obj):
	index = 1
	#creating material for the object
	ob = obj
	# Get material
	mat = bpy.data.materials.get("Material")
	if mat is None:
	    # create material
	    mat = bpy.data.materials.new(name="Material")
	# Assign it to object
	if ob.data.materials:
	    # assign to 1st material slot
	    ob.data.materials[0] = mat
	else:
	    # no slots
	    ob.data.materials.append(mat)
	#load image
	img = load_img(TX_PATH)
	#assigning texture
	mat.use_nodes = True
	node_tree = bpy.data.materials["Material"].node_tree
	#see if it already has an image
	node = None
	for n in node_tree.nodes:
		if n.type == 'TEX_IMAGE':
			node = n
	if node == None:
		node = node_tree.nodes.new("ShaderNodeTexImage")
	node.image = img

def create_light():
	# Create new lamp datablock
	lamp_data = bpy.data.lamps.new(name="New Lamp", type='POINT')
	# Create new object with our lamp datablock
	lamp_object = bpy.data.objects.new(name="New Lamp", object_data=lamp_data)
	# Link lamp object to the scene so it'll appear in this scene
	scene = bpy.context.scene
	scene.objects.link(lamp_object)
	# Place lamp to a specified location
	lamp_object.location = (5.0, 5.0, 5.0) # Light is always from the same position
	# And finally select it to make it active
	lamp_object.select = True
	scene.objects.active = lamp_object
	return lamp_object

def get_lights():
	lights_list = []
	#check if scene has lights
	count = 0
	for i in bpy.data.objects:
		if i.type=='LAMP':
			count += 1
			lights_list.append(i)
	#if we don't have enough light sources
	if count < LIGHTS_COUNT:
		#check how many we are missing
		count = LIGHTS_COUNT - count
		#create that many lights
		for c in range(count):
			i = create_light()
			lights_list.append(i)
	return lights_list

def delete_all_lights():
	bpy.ops.object.select_all(action='DESELECT')
	bpy.ops.object.select_by_type(type='LAMP')
	bpy.ops.object.delete()

def random_lights():
	global LIGHTS_COUNT
	#get existing lights
	lights_list = get_lights()
	#set random locations
	for i in range(len(lights_list)):
		rand_azimuth = np.random.uniform(0.0, 10)
		rand_elev = np.random.uniform(0.0, 1.0)
		rand_r = np.random.uniform(3, 6)
		lights_list[i].location = sph2cart(rand_azimuth, rand_elev, rand_r)

def camera_look_at(camera_obj, point):
    loc_camera = camera_obj.location
    direction = point - loc_camera
    # point the cameras '-Z' and use its 'Y' as up
    rot_quat = direction.to_track_quat('-Z', 'Y')
    # assume we're using euler rotation
    camera_obj.rotation_euler = rot_quat.to_euler()

def sph2cart(az, el, r):
    rcos_theta = r * np.cos(el)
    x = rcos_theta * np.cos(az)
    y = rcos_theta * np.sin(az)
    z = r * np.sin(el)
    return x, y, z

def random_camera():
	#find camera
	camera_obj = None
	for i in bpy.data.objects:
		if i.type=='CAMERA':
			camera_obj = i
	#set camera position
	rand_azimuth = np.random.uniform(0.0, 3.14*2)
	rand_elev = np.random.uniform(-1.57, 1.57) # +-pi*0.5
	rand_r = 2 
	camera_obj.location = sph2cart(rand_azimuth, rand_elev, rand_r) 
	
	yaw = rand_azimuth*180/np.pi
	pitch = rand_elev*180/np.pi 
	#set camera rotation
	camera_look_at(camera_obj, mathutils.Vector((0.0,0.0,0.0)))
	pos = [camera_obj.location.x, camera_obj.location.y, camera_obj.location.z, pitch, yaw]

	return pos

def render():
	bpy.data.scenes['Scene'].cycles.film_transparent = True
	bpy.data.scenes['Scene'].render.resolution_x = RES_X
	bpy.data.scenes['Scene'].render.resolution_y = RES_Y
	bpy.data.scenes['Scene'].render.resolution_percentage = 100
	bpy.data.scenes['Scene'].render.image_settings.file_format = 'JPEG'
	bpy.data.scenes['Scene'].render.image_settings.color_mode = 'BW'

	#uniform background color
	bpy.context.scene.world.horizon_color = (1, 1, 1)

def delete_meshes():
	bpy.ops.object.select_all(action='DESELECT')
	bpy.ops.object.select_by_type(type='MESH')
	bpy.ops.object.delete()

def clear_all_images():
	for block in bpy.data.images:
		bpy.data.images.remove(block)


def main():

	dir = DATASET_PATH #"/home/maxnihr/Documents/Dataset/ShapeNetCore.v2"
	file_name = FILE_NAME#"model_normalized.obj"
	outp_dir = OUTPUT_DIR#"/home/maxnihr/Documents/PreProData/"
	model_nr = 0
	# Following loop over model names is specific for the ShapeNet directory structure
	for name in glob.glob(dir+"/*/*/*/"+file_name):
		model_nr_id = "model_" + str(model_nr)
		model_nr += 1
		samples_per_model = 15
		# folder name indicates class in shapenet dataset, remove if not needed
		label = name.split("/")[6]
		out_put_filename = file_name
		make_dir = outp_dir + model_nr_id + "/"
		if os.path.exists(make_dir):
			continue
		else:
			os.mkdir(make_dir)

		obj= load_obj(name)
		#assign material
		add_material(obj)
		random_background()
		random_lights()
		render()

		for i in range(samples_per_model):
			model_nr_id_view = model_nr_id + "_" + str(i)
			pos = random_camera()

			#save camera view 
			file = open(make_dir + model_nr_id_view + ".txt", "w")
			pos.append(label)
			pos = '_'.join(str(e) for e in pos)
			file.write(pos) 
			file.close() 

			bpy.data.scenes['Scene'].render.filepath = make_dir + model_nr_id_view + '.jpg'
			bpy.ops.render.render( write_still=True )

		delete_meshes()
		clear_all_images()
	bpy.ops.wm.quit_blender()

main()


