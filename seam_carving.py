import numpy as np
from scipy.ndimage import sobel
from PIL import Image


class SeamCarver:
    def __init__(self, input_filename, new_height, new_width, protect_mask=None, object_mask=None):
        self.input_image = np.asarray(Image.open(input_filename).convert('RGB'), dtype=np.float64)
        self.original_height, self.original_width = self.input_image.shape[:2]
        self.new_height = new_height
        self.new_width = new_width
        self.protect_mask = None if protect_mask is None else np.asarray(Image.open(protect_mask).convert('L'))
        self.object_mask = None if object_mask is None else np.asarray(Image.open(object_mask).convert('L'))
        self.energy_map = np.zeros((self.input_image.shape[0], self.input_image.shape[1]), dtype=int)
        self.prepare_masks()

    def prepare_masks(self):
        if self.protect_mask is not None:
            self.protect_mask = np.array(self.protect_mask > 0, dtype=int)
        if self.object_mask is not None:
            self.object_mask = np.array(self.object_mask > 0, dtype=int)

    def calculate_energy_map(self):
        dx = sobel(self.input_image, axis=1, mode='constant')
        dy = sobel(self.input_image, axis=0, mode='constant')
        self.energy_map = np.sqrt(dx ** 2 + dy ** 2).sum(axis=2)
        if self.protect_mask is not None:
            self.energy_map[self.protect_mask > 0] += 1e6
        if self.object_mask is not None:
            self.energy_map[self.object_mask > 0] -= 1e6

    def find_seam(self):
        rows, cols = self.energy_map.shape
        seam = np.zeros(rows, dtype=int)
        cost = self.energy_map.copy()
        backtrack = np.zeros_like(cost, dtype=int)

        for i in range(1, rows):
            for j in range(cols):
                min_cost = cost[i - 1, j]
                backtrack[i, j] = j
                if j > 0 and cost[i - 1, j - 1] < min_cost:
                    min_cost = cost[i - 1, j - 1]
                    backtrack[i, j] = j - 1
                if j < cols - 1 and cost[i - 1, j + 1] < min_cost:
                    min_cost = cost[i - 1, j + 1]
                    backtrack[i, j] = j + 1
                cost[i, j] += min_cost

        seam[-1] = np.argmin(cost[-1])
        for i in range(rows - 2, -1, -1):
            seam[i] = backtrack[i + 1, seam[i + 1]]
        return seam

    def remove_seam(self, seam):
        rows, cols, _ = self.input_image.shape
        new_image = np.zeros((rows, cols - 1, 3), dtype=self.input_image.dtype)
        for i in range(rows):
            new_image[i, :, :] = np.delete(self.input_image[i, :, :], seam[i], axis=0)
        self.input_image = new_image

    def insert_seam(self, seam):
        rows, cols, _ = self.input_image.shape
        new_image = np.zeros((rows, cols + 1, 3), dtype=self.input_image.dtype)
        for i in range(rows):
            col = seam[i]
            for ch in range(3):
                if col == 0:
                    pixel_value = self.input_image[i, col, ch]
                else:
                    pixel_value = (self.input_image[i, col - 1, ch] + self.input_image[i, col, ch]) / 2
                new_image[i, :col, ch] = self.input_image[i, :col, ch]
                new_image[i, col, ch] = pixel_value
                new_image[i, col + 1:, ch] = self.input_image[i, col:, ch]
        self.input_image = new_image

    def carve_column(self, insert=False):
        self.calculate_energy_map()
        seam = self.find_seam()
        if insert:
            self.insert_seam(seam)
        else:
            self.remove_seam(seam)

    def carve_row(self, insert=False):
        self.input_image = np.rot90(self.input_image, 1, (0, 1))
        self.carve_column(insert)
        self.input_image = np.rot90(self.input_image, -1, (0, 1))

    def resize(self):
        if self.new_width > self.original_width:
            for _ in range(self.new_width - self.original_width):
                self.carve_column(insert=True)
        else:
            for _ in range(self.original_width - self.new_width):
                self.carve_column(insert=False)

        if self.new_height > self.original_height:
            for _ in range(self.new_height - self.original_height):
                self.carve_row(insert=True)
        else:
            for _ in range(self.original_height - self.new_height):
                self.carve_row(insert=False)

    def save_result(self, output_filename):
        result_image = Image.fromarray(np.uint8(self.input_image))
        result_image.save(output_filename)
