import os
import logging
from pathlib import Path

import numpy as np
import pyvips
import torch

logging.basicConfig()
logger = logging.getLogger("predict")
logger.setLevel(logging.INFO)

MODELS_FOLDER_PATH = Path("assets")

def to_gpu(inp, gpu=0):
    return inp.cuda(gpu, non_blocking=True)


def to_tensor(x):
    x = x.astype("float32") / 255
    return torch.from_numpy(x).permute(2, 0, 1)


def read_img(path):
    slide = pyvips.Image.new_from_file(str(path))
    region = pyvips.Region.new(slide).fetch(0, 0, slide.width, slide.height)
    return np.ndarray(
        buffer=region, dtype=np.uint8, shape=(slide.height, slide.width, 3)
    )


def get_tiles(img, tile_size, n_tiles, mode=0):
    h, w, _ = img.shape
    pad_h = (tile_size - h % tile_size) % tile_size + ((tile_size * mode) // 2)
    pad_w = (tile_size - w % tile_size) % tile_size + ((tile_size * mode) // 2)

    img = np.pad(
        img,
        [[pad_h // 2, pad_h - pad_h // 2], [pad_w // 2, pad_w - pad_w // 2], [0, 0]],
        constant_values=255,
    )
    img = img.reshape(
        img.shape[0] // tile_size, tile_size, img.shape[1] // tile_size, tile_size, 3
    )
    img = img.transpose(0, 2, 1, 3, 4).reshape(-1, tile_size, tile_size, 3)

    n_tiles_with_info = (
        img.reshape(img.shape[0], -1).sum(1) < tile_size ** 2 * 3 * 255
    ).sum()
    if len(img) < n_tiles:
        img = np.pad(
            img, [[0, n_tiles - len(img)], [0, 0], [0, 0], [0, 0]], constant_values=255
        )

    idxs = np.argsort(img.reshape(img.shape[0], -1).sum(-1))[:n_tiles]
    img = img[idxs]

    return img, n_tiles_with_info >= n_tiles


def concat_tiles(tiles, n_tiles, image_size):
    idxes = list(range(n_tiles))

    n_row_tiles = int(np.sqrt(n_tiles))
    img = np.zeros(
        (image_size * n_row_tiles, image_size * n_row_tiles, 3), dtype="uint8"
    )
    for h in range(n_row_tiles):
        for w in range(n_row_tiles):
            i = h * n_row_tiles + w

            if len(tiles) > idxes[i]:
                this_img = tiles[idxes[i]]
            else:
                this_img = np.ones((image_size, image_size, 3), dtype="uint8") * 255

            h1 = h * image_size
            w1 = w * image_size
            img[h1 : h1 + image_size, w1 : w1 + image_size] = this_img

    return img

class DS(torch.utils.data.Dataset):
    def __init__(self, images, root, n_tiles, tile_size):
        self.images = images
        self.root = root
        self.n_tiles = n_tiles
        self.tile_size = tile_size

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        item = self.images[index]
        logger.info("Preprocess on image: %s", item.filename)
        img = read_img(os.path.join(self.root, item.filename))

        img, _ = get_tiles(
            img,
            tile_size=self.tile_size,
            n_tiles=self.n_tiles,
        )
        img = concat_tiles(
            img,
            n_tiles=self.n_tiles,
            image_size=self.tile_size,
        )

        img = to_tensor(img)
        return img, item.filename

    @staticmethod
    def collate_fn(x):
        x, y = list(zip(*x))

        return torch.stack(x), y


def perform_inference(cj, images, data_root, progress):
    tile_sizes = [(36, 256), (64, 192), (144, 128),]
    progress_delta = (progress-10) / len(tile_sizes)

    all_preds = []
    batch_size = 1
    logits = torch.zeros((batch_size, 3), dtype=torch.float32, device="cuda")

    for n_tiles, tile_size in tile_sizes:
        progress += progress_delta

        progress_message = "Starting inference for Dataset with n_tiles: {}, tile size: {}".format(n_tiles, tile_size)
        logger.info(progress_message)
        cj.job.update(progress=progress, statusComment=progress_message) 

        # Loading model
        model_path = (MODELS_FOLDER_PATH / f'{tile_size}')
        logger.info("Reading models from %s", model_path)

        models = [
            torch.jit.load(str(p)).cuda().eval()
            for p in model_path.rglob("model_best.pt")
        ]

        # Dataset and preprocess
        ds = DS(images, data_root, n_tiles=n_tiles, tile_size=tile_size)
        loader = torch.utils.data.DataLoader(
            ds,
            batch_size=batch_size,
            # TODO: Test different num_workers
            num_workers=8, #number of subprocesses to use for data, 0 = main process
            shuffle=False,
            collate_fn=DS.collate_fn,
            pin_memory=True,
        )

        # Predictions step
        preds = []
        with torch.no_grad():
            for x, _ in loader:
                logger.info("Start inference for image : {}/{}".format(len(preds) + 1, len(images)))
                x = to_gpu(x)
                bs = len(x)

                logits.zero_()
                for model in models: 
                    logits[:bs] += model(x).sigmoid()

                logits /= len(models)
                preds.extend(logits.cpu().numpy())
                logger.info("Predictions: %s", preds)
        all_preds.append(preds)

    all_preds = np.array(all_preds).mean(0).sum(-1).round().astype("int")

    logger.info("Ending inference with predicted class: %s", all_preds)
    cj.job.update(progress=100, statusComment="Ending inference with predicted class: {}".format(all_preds)) 
    
    return all_preds

