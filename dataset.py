# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import glob


class Dataset:
    def __init__(self, scale=10.0, shuffle=False, data_dir=None,
    rf1_x=16, rf1_y=16, rf1_offset_x=5, rf1_offset_y=5, rf1_layout_x=3, rf1_layout_y=1,
    gauss_mask_sigma=0.4):
        self.rf1_size = (rf1_y, rf1_x)
        self.rf1_layout_size = (rf1_layout_y, rf1_layout_x)
        self.rf2_size = (rf1_y+(rf1_layout_y-1)*rf1_offset_y,
                         rf1_x+(rf1_layout_x-1)*rf1_offset_x)
        self.rf1_offset_x = rf1_offset_x
        self.rf1_offset_y = rf1_offset_y
        self.gauss_mask_sigma = gauss_mask_sigma
        
        self.load_images(scale, data_dir)

        if shuffle:
            indices = np.random.permutation(len(self.patches))
            self.patches = self.patches[indices]
        
        self.mask = self.create_gauss_mask(sigma=gauss_mask_sigma,
                                           width=rf1_x,
                                           height=rf1_y)

    # load all PNG images from the default or user-specified directory
    def load_images(self, scale, data_dir):
        images = []

        if data_dir == None:
            # Use images from the paper
            # directory location is relative to the current script location
            dir_prefix = os.path.dirname(os.path.realpath(__file__))
            dir_name = os.path.join(dir_prefix, "data", "images_rao")
        else:
            dir_name = data_dir
        
        file_names = sorted(glob.glob(os.path.join(dir_name, "*.png")))

        for i in file_names:
            image = cv2.imread(i)
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY).astype(np.float32)
            images.append(image)

        images = np.array(images)
        self.load_sub(images, scale)

    def create_gauss_mask(self, sigma=0.4, width=16, height=16):
        """ Create gaussian mask. """
        mu = 0.0
        x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
        d = np.sqrt(x**2+y**2)
        g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
        mask = g.reshape((-1))
        mask = mask / np.max(mask)
        return mask

    def load_sub(self, images, scale):
        rf2_x = self.rf2_size[1]
        rf2_y = self.rf2_size[0]

        self.images = images
        
        filtered_images = []
        for image in images:
            filtered_image = self.apply_DoG_filter(image)
            filtered_images.append(filtered_image)

        self.filtered_images = filtered_images

        w = images.shape[2]
        h = images.shape[1]

        size_w = w // rf2_x
        size_h = h // rf2_y

        patches = np.empty((size_h * size_w * len(images), rf2_y, rf2_x), dtype=np.float32)

        for image_index, filtered_image in enumerate(filtered_images):
            for i in range(size_w):
                for j in range(size_h):
                    x = rf2_x * i
                    y = rf2_y * j
                    patch = filtered_image[y:y+rf2_y, x:x+rf2_x]
                    # (16, 26)
                    # print(patch.shape)
                    index = size_w*size_h*image_index + j*size_w + i
                    patches[index] = patch

        patches = patches * scale
        self.patches = patches

    def get_images_from_patch(self, patch, use_mask=True):
        rf1_x = self.rf1_size[1]
        rf1_y = self.rf1_size[0]
        rf1_offset_x = self.rf1_offset_x
        rf1_offset_y = self.rf1_offset_y
        rf1_layout_x = self.rf1_layout_size[1]
        rf1_layout_y = self.rf1_layout_size[0]

        images = []
        for i in range(rf1_layout_x):
            for j in range(rf1_layout_y):
                x = rf1_offset_x * i
                y = rf1_offset_y * j
                # Apply gaussian mask
                image = patch[y:y+rf1_y, x:x+rf1_x].reshape([-1])
                if use_mask:
                    image = image * self.mask
                images.append(image)
        return images

    def get_images(self, patch_index):
        patch = self.patches[patch_index]
        return self.get_images_from_patch(patch)

    def apply_DoG_filter(self, image, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        """
        Apply difference of gaussian (DoG) filter detect edge of the image.
        """
        g1 = cv2.GaussianBlur(image, ksize, sigma1)
        g2 = cv2.GaussianBlur(image, ksize, sigma2)
        return g1 - g2

    def get_bar_images(self, is_short):
        patch = self.get_bar_patch(is_short)
        return self.get_images_from_patch(patch, use_mask=True)

    def get_bar_patch(self, is_short):
        """
        Get bar patch image for end stopping test.
        """
        bar_patch = np.ones((16,26), dtype=np.float32)
    
        if is_short:
            bar_width = 6
        else:
            bar_width = 24
        bar_height = 2
    
        for x in range(bar_patch.shape[1]):
            for y in range(bar_patch.shape[0]):
                if x >= 26/2 - bar_width/2 and \
                x < 26/2 + bar_width/2 and \
                y >= 16/2 - bar_height/2 and \
                y < 16/2 + bar_height/2:
                    bar_patch[y,x] = -1.0

        # Sete scale with stddev of all patch images.
        scale = np.std(self.patches)
        # Original scaling value for bar
        bar_scale = 2.0
        return bar_patch * scale * bar_scale
