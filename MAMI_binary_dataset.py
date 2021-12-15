import torch
import torch.utils.data as data
import torchvision.utils as utils
import torchvision
import torchvision.transforms as T

class MAMI_binary_dataset(data.Dataset):

    def __init__(self, text, image_path, text_processor, label, max_length=128):
        self.text_processor = text_processor
        self.text = text
        self.label = label
        self.max_length = max_length
        self.image_path = image_path


    def __getitem__(self, index):
        img = self.load_image(self.image_path[index])

        data = img.astype(np.float32)
        data = 255 * data
        img = data.astype(np.uint8)
        img = self.transform(img)

        return self.text_processor(self.text[index], padding="max_length", max_length=self.max_length, truncation=True, return_tensors='pt'), img, self.label[index]


    transform = T.Compose([
        T.ToPILImage(),
        T.Resize(224),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])])


    def load_image(self, filename):
        img = Image.open(filename)
        img.load()
        data = np.asarray(img, dtype="float32")
        return data

