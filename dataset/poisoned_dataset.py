import random
from typing import Callable, Optional

from PIL import Image
from torchvision.datasets import CIFAR10, MNIST
import os 


##########################################
# 根据自定义Trigger，完成TriggerHandler类
##########################################
class TriggerHandler(object):

    def __init__(self, 
                 trigger_path: str, 
                 trigger_size: int, 
                 trigger_label: int, 
                 img_width: int, 
                 img_height: int):
        ##########################################
        # 完成初始化方法（可根据需要调整参数）
        ##########################################
        self.trigger_path = trigger_path
        self.trigger_size = trigger_size
        self.trigger_label = trigger_label
        self.img_width = img_width
        self.img_height = img_height
        if os.path.exists(trigger_path):
            self.trigger_img = Image.open(trigger_path).convert("RGB")
            # 调整trigger大小
            self.trigger_img = self.trigger_img.resize((trigger_size, trigger_size))
        else: raise FileNotFoundError(f"trigger file {trigger_path} not found")


    def put_trigger(self, img: Image.Image) -> Image.Image:
        
        ##########################################
        # 完成trigger插入方法
        ##########################################
        if img.mode != "RGB":
            img = img.convert("RGB")

        #中心位置放置trigger
        
        x = (self.img_width - self.trigger_size) // 2
        y = (self.img_height - self.trigger_size) // 2
        position = (x, y)

        # 创建图像的副本
        poisoned_img = img.copy()
    	# 将trigger粘贴到图像上
        poisoned_img.paste(self.trigger_img, position)
        return poisoned_img

class CIFAR10Poison(CIFAR10):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height, self.channels = self.__shape_info__()

        ############################
        # 实例化一个TriggerHandler对象
        self.trigger_handler = TriggerHandler(args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        ############################
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

class MNISTPoison(MNIST):

    def __init__(
        self,
        args,
        root: str,
        train: bool = True,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = False,
    ) -> None:
        super().__init__(root, train=train, transform=transform, target_transform=target_transform, download=download)

        self.width, self.height = self.__shape_info__()
        self.channels = 1

        ############################
        # 实例化一个TriggerHandler对象
        self.trigger_handler = TriggerHandler( args.trigger_path, args.trigger_size, args.trigger_label, self.width, self.height)
        ############################
        self.poisoning_rate = args.poisoning_rate if train else 1.0
        indices = range(len(self.targets))
        self.poi_indices = random.sample(indices, k=int(len(indices) * self.poisoning_rate))
        print(f"Poison {len(self.poi_indices)} over {len(indices)} samples ( poisoning rate {self.poisoning_rate})")

    @property
    def raw_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "raw")

    @property
    def processed_folder(self) -> str:
        return os.path.join(self.root, "MNIST", "processed")


    def __shape_info__(self):
        return self.data.shape[1:]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode="L")
        # NOTE: According to the threat model, the trigger should be put on the image before transform.
        # (The attacker can only poison the dataset)
        if index in self.poi_indices:
            target = self.trigger_handler.trigger_label
            img = self.trigger_handler.put_trigger(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

