import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import torch

from dataset import MiniImageNetDataset

def collate_fn(batch):
    imgs = [item['image'] for item in batch if item['image'] is not None]
    targets = [item['label'] for item in batch if item['image'] is not None]
    filenames = [item['filename'] for item in batch if item['image'] is not None]
    imgs = torch.stack(imgs)
    targets = torch.stack(targets)
    return {'image': imgs, 'label': targets, 'filename': filenames}

def get_loader(args):
    transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((args.image_size, args.image_size)),
            transforms.ToTensor(),
            transforms.Normalize([0.5]*len(args.diff_channel), [0.5]*len(args.diff_channel))
        ])
    
    train_data = MiniImageNetDataset(args.root_dir, args.train_file_path, transform = transform, image_size=args.image_size, diff_channel=args.diff_channel)
    train_loader = DataLoader(dataset = train_data, batch_size = args.batch_size, shuffle=True, num_workers=8, collate_fn=collate_fn)

    val_data = MiniImageNetDataset(args.root_dir, args.val_file_path, transform = transform, image_size=args.image_size, diff_channel=args.diff_channel)
    val_loader = DataLoader(dataset = val_data, batch_size = args.batch_size, shuffle=True, collate_fn=collate_fn)

    test_data = MiniImageNetDataset(args.root_dir, args.test_file_path, transform = transform, image_size=args.image_size, diff_channel=args.diff_channel)
    test_loader = DataLoader(dataset = test_data, batch_size = args.batch_size, shuffle = True, collate_fn=collate_fn)
    
    return train_data, val_data, test_data, train_loader, val_loader, test_loader
