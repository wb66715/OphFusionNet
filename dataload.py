import os
from torch.utils.data import Dataset
from PIL import Image


class MultiModalDataset(Dataset):
    def __init__(self, image_folder_path, txt_file_path, transform=None):
        self.image_folder_path = image_folder_path
        self.txt_file_path = txt_file_path
        self.transform = transform
        self.pairs = self.read_pair_file()
        self.image_formats = ['.jpg', '.jpeg', '.png', '.bmp', '.TIF']  # 支持的格式

    def read_pair_file(self):
        pairs = []
        with open(self.txt_file_path, 'r', encoding='utf-8-sig') as file:
            lines = file.readlines()
            for line in lines:
                line = line.strip().split()
                if len(line) == 3:
                    # 移除潜在的 BOM 字符
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

