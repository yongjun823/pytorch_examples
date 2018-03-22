import torch
import torch.utils.data as data
import torchvision.transforms as transforms
from PIL import Image
import os

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

pre_process = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])


# You should build custom dataset as below.
class CustomDataset(data.Dataset):
    def __init__(self, db_path):
        self.path = os.path.join(os.getcwd(), db_path)
        self.image_names = os.listdir(db_path)
        self.labels = [name[0] for name in self.image_names]
        self.image_labels = set(self.labels)

    def __getitem__(self, index):
        img_name = self.image_names[index]
        img_label = img_name[0]

        im = Image.open(os.path.join(self.path, img_name))
        img = pre_process(im)

        return img, img_label

    def __len__(self):
        return len(self.image_names)

        # Then, you can just use prebuilt torch's data loader.


custom_dataset = CustomDataset('tt')
train_loader = torch.utils.data.DataLoader(dataset=custom_dataset,
                                           batch_size=2,
                                           shuffle=True)

for image, label in train_loader:
    print(label)
