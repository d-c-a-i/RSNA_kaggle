# import torchvision


# def get_transforms(aug=False):
#     """
#     # old transforms
#     create_transform(
#         (1024, 512), 
#         mean=0.53, #(0.53, 0.53, 0.53),
#         std=0.23, #(0.23, 0.23, 0.23),
#         is_training=is_training, 
#         auto_augment=f'rand-m{config.AUTO_AUG_M}-n{config.AUTO_AUG_N}'
#     )
#     """
#     def transforms(img):
#         img = img.convert('RGB')#.resize((512, 512))
#         if aug:
#             tfm = [
#                 torchvision.transforms.RandomHorizontalFlip(0.5),
#                 torchvision.transforms.RandomRotation(degrees=(-5, 5)), 
#                 torchvision.transforms.RandomResizedCrop((1024, 512), scale=(0.8, 1), ratio=(0.45, 0.55)) 
#             ]
#         else:
#             tfm = [
#                 torchvision.transforms.RandomHorizontalFlip(0.5),
#                 torchvision.transforms.Resize((1024, 512))
#             ]
#         img = torchvision.transforms.Compose(tfm + [            
#             torchvision.transforms.ToTensor(),
#             torchvision.transforms.Normalize(mean=0.2179, std=0.0529),
            
#         ])(img)
#         return img

#     return lambda img: transforms(img)