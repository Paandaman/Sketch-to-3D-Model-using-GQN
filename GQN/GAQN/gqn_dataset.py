from torch.utils.data import Dataset
import glob
import tensorflow as tf

class GqnDataset(Dataset):
    # Loads data from the GQN Shepard-Metzler dataset
    # in tfrecords format and converts it to numpy format
    # (dataset can be found at https://github.com/deepmind/gqn-datasets)

    def __init__(self, root_dir, transform=None):
        self.root = root_dir
        self.file_names = self.get_names(root_dir)
        self.pos_dim = 5
        self.seq_length = 15
        self.image_size = 64

    def __len__(self):
        return len(self.get_names(self.root))

    def get_names(self, dir):
        file_names = []
        for name in glob.glob(dir+"/*.tfrecord"):
            file_names.append(name)
        return file_names

    def __getitem__(self, idx):
        batch_name = self.file_names[idx]
        batch = self.load_data(batch_name)
        return batch

    def load_data(self, file_name):
        # Each file contains a list of tuples, where each tuple contains (images, poses)
        data = self.convert_record(file_name)
        return data

    def convert_record(self, record, batch_size=None):
        scenes = self.process_record(record, batch_size)
        return scenes

    def process_record(self, record, batch_size=None):
        engine = tf.python_io.tf_record_iterator(record)
        scenes = []
        for i, data in enumerate(engine):
            if i == batch_size:
                break
            scene = self.convert_to_numpy(data)
            scenes.append(scene)

        return scenes

    def process_images(self, example):
        images = tf.concat(example['frames'], axis=0)
        images = tf.map_fn(tf.image.decode_jpeg, tf.reshape(images, [-1]),
                           dtype=tf.uint8, back_prop=False)
        shape = (self.image_size, self.image_size, 3)
        images = tf.reshape(images, (-1, self.seq_length) + shape)
        images = tf.cast(images, dtype=tf.float32)
        OldMin = 0
        OldMax = 255
        NewMin = -1
        NewMax = 1
        OldRange = (OldMax - OldMin)  
        NewRange = (NewMax - NewMin)
        images = tf.add(tf.div(tf.multiply(tf.subtract(images, OldMin), NewRange), OldRange), NewMin)
        return images

    def process_poses(self, example):
        poses = example['cameras']
        poses = tf.reshape(poses, (-1, self.seq_length, self.pos_dim))
        return poses

    def convert_to_numpy(self, raw_data):

        feature = {'frames': tf.FixedLenFeature(shape=self.seq_length, dtype=tf.string),
                   'cameras': tf.FixedLenFeature(shape=self.seq_length * self.pos_dim, dtype=tf.float32)}
        example = tf.parse_single_example(raw_data, feature)

        images = self.process_images(example)
        poses = self.process_poses(example)
        return images.numpy().squeeze(), poses.numpy().squeeze()

