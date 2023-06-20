import os
import random
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path


class ImageLoader:
    def __init__(self, directory, transform):
        self.directory = directory
        self.images = self.load_images()
        self.transform = transform

    def load_images(self):
        image_list = []
        for root, dirs, files in os.walk(self.directory):
            for file in files:
                if (
                    file.endswith(".jpg")
                    or file.endswith(".jpeg")
                    or file.endswith(".png")
                    or file.endswith(".JPEG")
                ):
                    image_path = os.path.relpath(
                        os.path.join(root, file), self.directory
                    )
                    image_list.append(image_path)
        return image_list

    def get_random_input(self):
        random_image_path = random.choice(self.images)
        image, image_path = self.load_image(random_image_path)
        return image, image_path

    def get_random_inputs(self, num_of_inputs):
        random_image_paths = random.sample(self.images, num_of_inputs)
        images = []
        for random_image_path in random_image_paths:
            image, image_path = self.load_image(random_image_path)
            images.append((image, image_path))
        return images

    def load_image(self, image_path):
        image = Image.open(os.path.join(self.directory, image_path))
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, image_path

    def get_image_from_path(self, image_path):
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, image_path
