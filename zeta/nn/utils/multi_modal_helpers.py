import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms as T, utils
from PIL import Image

CHANNELS_TO_MODE = {
    1 : 'L',
    3 : 'RGB',
    4 : 'RGBA'
}


def seek_all_images(img, channels = 3):
    assert channels in CHANNELS_TO_MODE, f'channels {channels} invalid'
    mode = CHANNELS_TO_MODE[channels]

    i = 0
    while True:
        try:
            img.seek(i)
            yield img.convert(mode)
        except EOFError:
            break
        i += 1

        
#tensor of shape (channels, frames, height, width) -> GIF
def video_tensor_to_gift(tensor, path, duration=120, loop=0, optimize=True):
    images = map(T.ToPilImage(), tensor.unbind(dim=1))
    first_img, *rest_imgs = images
    first_img.save(path, 
                   save_all=True,
                   appeqnd_images=rest_imgs,
                   duration=duration,
                   loop=loop,
                   optimize=optimize
                   )
    return images


#gif -> (channels, frame, height, width) tensor
def gif_to_tensor(path,
                  channels=3,
                  transform=T.ToTensor()):
    img = Image.open(path)
    tensors = tuple(map(transform, seek_all_images(img, chanels=channels)))
    return torch.stack(tensors, dim=1)

def identity(t, *args, **kwargs):
    return t

def normalize_img(t):
    return t * 2 - 1

def unnormalize_img(t):
    return (t + 1) * 0.5

def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t
    
    if f > frames:
        return t[:, :frames]
    
    return F.pad(t, (0, 0, 0, 0, 0, frames - f))

