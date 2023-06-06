from torchvision import transforms
import torch
from PIL import Image
from segmentation.unet import UNet, Encoder, Decoder, MainConv
import cv2
import os


class Segmentator:
    def __init__(self, project_folder_path):
        self.classic_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(512)
        ])

        self.dir_path = project_folder_path

        self.model = torch.load(f'{self.dir_path}\\segmentation\\unet_test_3_024',
                                map_location=torch.device('cpu'))

    def segment(self, image):
        img = self.classic_transform(image)
        dir_path_correct = self.dir_path.replace('\\', '/')
        # print(cv2.imwrite(f'{dir_path_correct}/static/before_updated_image.png', img.numpy()))
        # print(img.size())

        output = self.model(img.unsqueeze(0)).detach()
        output = output[0]

        print(image.size)
        output = transforms.CenterCrop((image.size[1], image.size[0]))(output)
        print(output.size())

        numpy_array = output.permute(1, 2, 0).numpy()
        numpy_array *= -170

        return cv2.imwrite(f'{dir_path_correct}/static/updated_image.png', numpy_array)


if __name__ == "__main__":
    s = Segmentator(os.getcwd()[:-13])
    image_path = f'{os.getcwd()[:-13]}\\static\\image.png'
    img = Image.open(image_path)
    print(img)
    print(s.segment(img))
