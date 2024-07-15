import os
import random
import torch
from torchvision import transforms
from PIL import Image
from pathlib import Path

"""
Overview: The purpose of this class is to make the data set to be used for 
the execution and analysis of the model during the execution of the program 
suitable for the model, to prepare sample input and to make it ready for the 
use of other classes.

Maintainers: - Osman Çağlar - cglrr.osman@gmail.com
             - Abdul Hannan Ayubi - abdulhannanayubi38@gmail.com
"""


class ImageLoader:
    def __init__(self, directory, transform):
        self.directory = directory
        self.images = self.load_images()
        self.transform = transform

    def load_images(self):
        """
        This function is used to load the images in the given directory.

        Args:
            None
        Returns:
            image_list (list): List of images in the given directory.
        """
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
        """
        This function is used to return random image from the loaded images.

        Args:
            None
        Returns:
            image (tensor): Random image from the loaded images.
            image_path (str): Path of the random image.
        """
        random_image_path = random.choice(self.images)
        image, image_path = self.load_image(random_image_path)
        return image, image_path

    def get_random_inputs(self, num_of_inputs):
        """
        This function is used to return random images from the loaded images.

        Args:
            num_of_inputs (int): Number of random images to be returned.
        Returns:
            images (list): List of random images from the loaded images.
        """
        random_image_paths = random.sample(self.images, num_of_inputs)
        images = []
        for random_image_path in random_image_paths:
            image, image_path = self.load_image(random_image_path)
            images.append((image, image_path))
        return images

    def load_image(self, image_path):
        """
        This function is used to load the image in the given path. Then, applies the
        given transformation as parameter to the image and returns the image tensor and
        the path of the image.

        Args:
            image_path (str): Path of the image to be loaded.
        Returns:
            image_tensor (tensor): Image tensor of the loaded image.
            image_path (str): Path of the loaded image.
        """
        image = Image.open(os.path.join(self.directory, image_path))
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, image_path

    def get_image_from_path(self, image_path):
        """
        This function is used to load the image in the given path. Then, applies the
        given transformation as parameter to the image and returns the image tensor and
        the path of the image.

        Args:
            image_path (str): Path of the image to be loaded.
        Returns:
            image_tensor (tensor): Image tensor of the loaded image.
            image_path (str): Path of the loaded image.
        """
        image = Image.open(image_path)
        image_tensor = self.transform(image).unsqueeze(0)
        return image_tensor, image_path
