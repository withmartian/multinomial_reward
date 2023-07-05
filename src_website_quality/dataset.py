from PIL import Image
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

def resize_and_slice(img_path, desired_size=224, fill_color=(255, 255, 255)):
    img = Image.open(img_path)
    ratio = desired_size / img.width
    new_height = int(ratio * img.height)
    img = img.resize((desired_size, new_height), Image.ANTIALIAS)

    image_slices = []

    # If the new height is less than the desired size, pad it
    if new_height < desired_size:
        new_img = Image.new("RGB", (desired_size, desired_size), fill_color)
        new_img.paste(img, (0,0))
        image_slices.append(new_img)
    else:
        for i in range(0, new_height, desired_size):
            if i + desired_size <= new_height:
                new_img = img.crop((0, i, desired_size, i + desired_size))
                # convert to RGB format
                new_img = new_img.convert("RGB")
                image_slices.append(new_img)

    return image_slices


class ImageRankingDataset(Dataset):
  # rankings contains lists of image file paths of the same website
  # [[wiki2001.png, wiki2002.png... ], [baidu2007.png, baidu2008.png...], ...]
  # 1997 - 2023 is 27 years
  # self.items eventually is a list of tokenized rankings for each website with padding
  def __init__(self, rankings, processor, max_ranks_per_batch=27, img_width=224, img_height=224):
    self.items = []
    for ranking in tqdm(rankings): # for each website
      images = [resize_and_slice(path) for path in ranking] # a list of slices
      slice_count = [len(sublist) for sublist in images]
      flattened_imgs = [img for sublist in images for img in sublist]
      inputs = processor(images=flattened_imgs, return_tensors="pt", padding=True)
      item = {"pixel_values": inputs['pixel_values'], "counts": [len(ranking), slice_count]}
      self.items.append(item)

  def __len__(self):
      return len(self.items)

  def __getitem__(self, idx):
      return self.items[idx]
  

class DataCollator:
  def __init__(self, max_ranks_per_batch=27):
    self.max_ranks_per_batch = max_ranks_per_batch

  def __call__(self, batch): # batch here is a list of items from ImageRankingDataset
    batch_data = {}
    batch_data["pixel_values"] = torch.cat([item["pixel_values"] for item in batch])
    batch_data["counts"] = [item["counts"] for item in batch]
    batch_data["labels"] = torch.tensor([i for i in range(self.max_ranks_per_batch) for _ in batch])
    return batch_data