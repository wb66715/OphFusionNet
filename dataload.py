import os
import random
import numpy as np
from PIL import Image
import cv2
from torch.utils.data import Dataset


class MultiModalDataset(Dataset):
    def __init__(self, image_folder_path, txt_file_path, transform=None, apply_gaussian_noise=False,
                 apply_color_transfer=False):
        self.image_folder_path = image_folder_path
        self.CFPimage_folder_path = "./gamma分级/gamma/cfptrain"   #   self.CFPimage_folder_path = "./cfp图片"   #

        self.txt_file_path = txt_file_path
        self.transform = transform
        self.apply_gaussian_noise = apply_gaussian_noise  # Flag to indicate if noise should be added
        self.apply_color_transfer = apply_color_transfer  # Flag to indicate if color transfer should be applied
        self.pairs = self.read_pair_file()
        self.image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.TIF']  # Supported formats

    def read_pair_file(self):
        pairs = []
        with open(self.txt_file_path, 'r', encoding='utf-8-sig') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) == 3:
                    img1_name = line[0].lstrip('\ufeff')
                    img2_name = line[1].lstrip('\ufeff')
                    pair = (img1_name, img2_name, int(line[2]))
                    pairs.append(pair)
        return pairs

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        img1_name, img2_name, label = self.pairs[idx]

        img1_path = self.get_image_path(img1_name)
        img2_path = self.get_image_path(img2_name)

        img1 = Image.open(img1_path).convert('RGB')
        img2 = Image.open(img2_path).convert('RGB')

        # If color transfer is enabled, apply color transfer to img1
        if self.apply_color_transfer:
            # Randomly select another image from the color images for color transfer
            color_images = [f for f in os.listdir(self.CFPimage_folder_path) if
                            f.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.TIF'))]
            color_images = [f for f in color_images if f != img1_name]  # Exclude img1 from the list

            random_color_img_name = os.path.splitext(random.choice(color_images))[0]

            random_color_img_path = self.get_image_path(random_color_img_name)

            random_color_img = Image.open(random_color_img_path).convert('RGB')

            # Apply color transfer only to img1 using the randomly selected color image
            img1 = self.apply_color_transfer_func(img1, random_color_img)  # Only img1 is modified, img2 is ignored

        if self.add_gaussian_noise:
            img2 = self.add_gaussian_noise(img2)

        if self.transform:
            img1 = self.transform(img1)
            img2 = self.transform(img2)

        return img1, img2, label

    def get_image_path(self, img_name):
        for ext in self.image_formats:
            img_path = os.path.join(self.image_folder_path, f'{img_name}{ext}')
            if os.path.isfile(img_path):
                return img_path
        raise FileNotFoundError(f'未找到支持格式的图像: {img_name}')

    def add_gaussian_noise(self, image, mean=None, std=None):
        """ Adds Gaussian noise to the image. Mean and std can be random. """
        if mean is None:
            mean = random.uniform(0, 10)  # Random mean
        if std is None:
            std = random.uniform(0, 20)  # Random standard deviation

        # Convert the image to a numpy array
        image_array = np.array(image)

        # Generate Gaussian noise
        gauss = np.random.normal(mean, std, image_array.shape)

        # Add noise to the image
        noisy_image = np.clip(image_array + gauss, 0, 255)  # Ensure pixel values are in [0, 255]

        return Image.fromarray(noisy_image.astype('uint8'))

    def apply_color_transfer_func(self, img1, img2):

        img1 = img1.resize((448, 448))
        img2 = img2.resize((448, 448))

        # Convert Image1 and Image2 from RGB to LAB
        img1 = np.array(img1)
        img2 = np.array(img2)

        lab1 = cv2.cvtColor(img1, cv2.COLOR_RGB2LAB)
        lab2 = cv2.cvtColor(img2, cv2.COLOR_RGB2LAB)

        # Calculate mean and std of LAB images
        mean1, stddev1 = cv2.meanStdDev(lab1)
        mean2, stddev2 = cv2.meanStdDev(lab2)

        # Convert mean and std to 1D arrays for easier broadcasting
        mean1 = mean1.flatten()
        stddev1 = stddev1.flatten()
        mean2 = mean2.flatten()
        stddev2 = stddev2.flatten()

        # Adjust Lab1 to match Lab2's color distribution
        lab1_new = (lab1 - mean1) / stddev1 * stddev2 + mean2

        # Clip pixel values to ensure they are in valid range (0-255)
        np.putmask(lab1_new, lab1_new > 255, 255)
        np.putmask(lab1_new, lab1_new < 0, 0)

        # Convert LAB back to RGB
        new_image1 = cv2.cvtColor(np.uint8(lab1_new), cv2.COLOR_LAB2RGB)

        # Convert numpy arrays back to PIL images
        new_image1 = Image.fromarray(new_image1)

        return new_image1
