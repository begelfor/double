import cv2
import time
import numpy as np
from tqdm import tqdm
from collections import defaultdict
from projective_plane import get_lines
from generate_image import ImageGenerator
from utils import rotate_and_scale
from pathlib import Path


class Double:
    def __init__(self, folder:str, word_list:list[str], n:int, card_size: int,
                 min_distance: int = 10,
                 C: float = 0.2,
                 n_tries: int = 30,
                 k_tries: int = 30):
        self.folder = Path(folder)
        self.card_size = card_size
        self.min_distance = min_distance
        self.C = C
        self.n_tries = n_tries
        self.k_tries = k_tries
        self.image_folder = self.folder / "images"
        if not self.image_folder.exists():
            self.image_folder.mkdir(parents=True)
        self.card_folder = self.folder / "cards"
        if not self.card_folder.exists():
            self.card_folder.mkdir(parents=True)
        lines = get_lines(n)
        if not len(lines) == len(word_list):
            raise ValueError("Number of words does not match number of lines")
        self.word_list = word_list
        # self.create_images()
        self.word2imgs = self.get_images_and_masks()

        self.tries_stats = defaultdict(list)
        for i, line in tqdm(enumerate(lines)):
            card = self.create_card([word_list[i] for i in line])
            cv2.imwrite(self.folder/"cards" / f"{i}.jpg", card)
        for key, l in self.tries_stats.items():
            print(f"{key}: {sum(l)/len(l):1.1f}")


    def create_images(self):
        image_generator= ImageGenerator()
        existing_images = self.image_folder.glob("*.jpg")
        already_done = [x.stem for x in existing_images]

        for i, word in tqdm(enumerate(set(self.word_list) - set(already_done))):
            if i>0:
                time.sleep(40)
            prompt = f" A comics style image of {word} with a white background"
            path  = image_generator.generate_image_from_text(prompt, self.image_folder/word)

            img = cv2.imread(path)
            # Find non-white pixels
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
            
            # Find bounding box of non-white pixels
            coords = cv2.findNonZero(binary)
            x, y, w, h = cv2.boundingRect(coords)
            
            # Crop the image
            img = img[y:y+h, x:x+w]
            cv2.imwrite(path, img)

    def get_images_and_masks(self):
        d = {}
        for path in self.image_folder.glob("*.jpg"):
            if path.stem in self.word_list:
                word = path.stem
                img = cv2.imread(path)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                _, binary = cv2.threshold(gray, 250, 255, cv2.THRESH_BINARY_INV)
                d[word] = {"image":img, "mask":binary}
        return d

    def create_card(self, card_words:list[str]):
        words = set(card_words)
        # we first work only with masks
        card = np.zeros((self.card_size, self.card_size), dtype=np.uint8)
        thickness = 2

        card = cv2.circle(card, (self.card_size//2, self.card_size//2), self.card_size//2, 255, -1)
        cv2.floodFill(card, None, seedPoint=(0,0), newVal=0)
        remove_card_counter = 0

        params = {}
        while len(params) < len(card_words):
            distance_matrix = cv2.distanceTransform(card, cv2.DIST_L2, 5)
            # Flatten distance matrix and create probability distribution
            flat_distances = distance_matrix.flatten()
            # Add small constant to avoid division by zero
            probs = flat_distances + 1e-10
            probs = probs / probs.sum()
            for i_n in range(self.n_tries):
                word = np.random.choice(list(words-set(params.keys())))
                scale = np.random.uniform(.2, .5)
                angle = np.random.uniform(0, 360)
                mask = rotate_and_scale(self.word2imgs[word]["mask"], scale=scale, angle=angle)
                y_cm, x_cm = np.array(np.where(mask>0)).mean(axis=1).astype(int)
                height, width = mask.shape
                # Sample random index according to distance probabilities
                for i_k in range(self.k_tries):
                    idx = np.random.choice(len(probs), p=probs)
                    # Convert back to 2D coordinates
                    y, x = np.unravel_index(idx, distance_matrix.shape)
                    if y-y_cm < 0 or x-x_cm < 0 or y-y_cm + height> self.card_size or x-x_cm + width> self.card_size:
                        continue

                    all_distances = distance_matrix[y-y_cm:y-y_cm+height, x-x_cm:x-x_cm+width][mask>0]
                    if all_distances.min() < self.min_distance:
                        continue
                    params[word] = {"scale":scale, "angle":angle, "offset_y":y-y_cm, "offset_x":x-x_cm}
                    card[y-y_cm:y-y_cm+height, x-x_cm:x-x_cm+width][mask>0] = 0
                    break
                else:
                    continue
                self.tries_stats[len(params)].append(i_n+1)
                break
            else:
                # remove a random word
                if len(params) == 0:
                    raise RuntimeError("Failed to create card")
                word = np.random.choice(list(params.keys()))
                mask = rotate_and_scale(self.word2imgs[word]["mask"], params[word]["angle"], params[word]["scale"])
                offset_y = params[word]["offset_y"]
                offset_x = params[word]["offset_x"]
                height, width = mask.shape
                card[offset_y:offset_y+height, offset_x:offset_x+width][mask>0] = 255
                params.pop(word)
                remove_card_counter += 1
                cv2.imwrite(self.folder/"cards" / f"{'_'.join(card_words)}_{remove_card_counter}.jpg", card)
        # now we create the actual card
        card = np.full((self.card_size, self.card_size, 3), 255, dtype=np.uint8)
        cv2.circle(card, (self.card_size//2, self.card_size//2), self.card_size//2-thickness, (0, 0, 0), thickness)
        for word in params:
            img = self.word2imgs[word]["image"]
            img = rotate_and_scale(img, params[word]["angle"], params[word]["scale"])
            mask = self.word2imgs[word]["mask"]
            mask = rotate_and_scale(mask, params[word]["angle"], params[word]["scale"])
            offset_y = params[word]["offset_y"] 
            offset_x = params[word]["offset_x"]
            height, width, _ = img.shape
            card[offset_y:offset_y+height, offset_x:offset_x+width][mask>0] = img[mask>0]
        return card

def main():
    folder = '/home/evg/Data/double'
    word_list = [
    'ice cream', 'donut', 'cupcake', 'salad', 'cake', 'pancake', 'waffle', 'apple', 
    'watermelon', 'cat', 'dog', 'elephant', 'fish', 'horse', 'tiger', 'mouse', 'rabbit',
    'bird', 'snake', 'frog', 'turtle', 'monkey', 'duck', 'donkey', 'ant', 'alligator',
    'bear', 'ball', 'hat', 'bag', 'sunglasses', 'umbrella', 'shoe', 'shirt', 'pants', 'dress', 
    'car', 'window', 'door', 'cup', 'computer', 'sun', 'star', 'toothbrush', 'plate',
    'bowl', 'fork', 'knife', 'spoon', 'bottle', 'hand', 'leg', 'head', 'eye', 'nose', 'sink',
    'potato', 'carrot', 'pineapple', 'couch', 'chair', 'table', 'bed', 'lamp',
    'mirror', 'clock', 'key', 'book', 'pen', 'pencil', 'paper', 'blanket', 'pillow']
    n = 7
    L = n*n+n+1
    word_list = word_list[:L]
    double = Double(folder=folder, word_list=word_list, n=n, card_size=1000)

if __name__ == "__main__":
    main()
