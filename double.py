import cv2
import glob
import os
import time
import numpy as np
from tqdm import tqdm

from projective_plane import get_lines
from generate_image import generate_image_from_text
from utils import rotate_and_scale
from pathlib import Path

class Double:
    def __init__(self, folder:str, word_list:list[str], n:int, card_size: int):
        self.folder = Path(folder)
        self.card_size = card_size
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
        self.create_images()
        self.word2imgs = self.get_images_and_masks()
        self.cards_words = [[word_list[i] for i in line] for line in lines]

    def create_images(self):
        existing_images = self.image_folder.glob("*.jpg")
        already_done = [x.stem for x in existing_images]

        for i, word in tqdm(enumerate(set(self.word_list) - set(already_done))):
            if i>0:
                time.sleep(40)
            prompt = f" A comics style image of {word} with a white background"
            path  = generate_image_from_text(prompt, self.image_folder/word)

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
        card = np.ones((self.card_size, self.card_size), dtype=np.uint8)
        thickness = 4
        cv2.circle(card, (self.card_size//2, self.card_size//2), self.card_size//2-thickness, 0, thickness)

        word2parameters = {}
        n_tries = 10
        k_tries = 10
        while True:
            if len(word2parameters) == len(card_words):
                # remove a random word
                word = np.random.choice(list(word2parameters.keys()))
                mask = rotate_and_scale(self.word2imgs[word]["mask"], **word2parameters[word])
                card = cv2.bitwise_xor(card, mask) #TODO pick the right operation
                word2parameters.pop(word)

            distance_matrix = cv2.distanceTransform(card, cv2.DIST_L2, 5)
            for _ in range(n_tries):
                word = np.random.choice(list(words-set(word2parameters.keys())))
                scale = np.random.uniform(.2, .6)
                angle = np.random.uniform(0, 360)
                mask = rotate_and_scale(self.word2imgs[word]["mask"], scale=scale, angle=angle)
                height, width = mask.shape
                for _ in range(k_tries):
                    offset_x = np.random.randint(0, self.card_size-width)
                    offset_y = np.random.randint(0, self.card_size-height)
                    distance = distance_matrix[offset_y:offset_y+height, offset_x:offset_x+width][mask>0].max()
                    


                

        











