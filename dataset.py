import os
import gzip
import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.transforms import ToTensor, Normalize, Compose

def read_idx(filename):
    """Reads MNIST idx file format into numpy array."""
    with gzip.open(filename, 'rb') as f:
        zero, data_type, dims = ord(f.read(1)), ord(f.read(1)), ord(f.read(1))
        num_dims = ord(f.read(1))
        shape = tuple(int.from_bytes(f.read(4), 'big') for _ in range(num_dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

class MNIST(VisionDataset):

    def __init__(self, root, train=True, transform=None, target_transform=None):
        super(MNIST, self).__init__(root, transform=transform,
                                           target_transform=target_transform)
        self.train = train  # training set or test set

        if self.train:
            self.image_file = '/content/data/train-images-idx3-ubyte.gz'
            self.label_file = '/content/data/train-labels-idx1-ubyte.gz'
        else:
            self.image_file = '/content/data/t10k-images-idx3-ubyte.gz'
            self.label_file = '/content/data/t10k-labels-idx1-ubyte.gz'

        self.image_path = os.path.join(root, self.image_file)
        self.label_path = os.path.join(root, self.label_file)

        # Default transformation pipeline
        self.transform = Compose([
            ToTensor(),  # Converts array to Tensor and scales [0, 255] -> [0.0, 1.0]
            Normalize(mean=[0.1307], std=[0.3081])  # Normalize the data
        ]) if transform is None else transform

        # Load data
        self.data = read_idx(self.image_path)
        self.targets = read_idx(self.label_path)

    def __len__(self):
        """Return the total number of images in the dataset."""
        return len(self.data)

    def __getitem__(self, index):
        """Return one item from the dataset."""
        img = Image.fromarray(self.data[index])
        target = self.targets[index]
        if self.transform is not None:
            img = self.transform(img)
        return img, target

if __name__ == "__main__":
    dataset = MNIST(root='.', train=True)
    print(dataset[0])  # Print the first (image, label) tuple
