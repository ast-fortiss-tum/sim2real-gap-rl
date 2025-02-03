import tensorflow as tf
import tensorflow_datasets as tfds

# Load the CycleGAN dataset (horse2zebra)
dataset = tfds.load('cycle_gan/horse2zebra', split='trainA')

# Example: show the first sample in the dataset
for example in dataset.take(1):
    image = example['image']
    print("Image shape:", image.shape)
    print("Image dtype:", image.dtype)

