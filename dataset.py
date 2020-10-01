# -*- coding: utf-8 -*-
import numpy as np
import cv2
import os
import glob


class Dataset:
    def __init__(self, scale=10.0, shuffle=False, data_dir=None,
                 rf1_x=16, rf1_y=16, rf1_offset_x=5, rf1_offset_y=5, rf1_layout_x=3, rf1_layout_y=1,
                 use_mask=True, gauss_mask_sigma=0.4,
                 image_filter="DoG", DoG_ksize=(5,5), DoG_sigma1=1.3, DoG_sigma2=2.6):
        # receptive field dimensions of a level-1 module (there are multiple level-1 modules)
        self.rf1_size = (rf1_y, rf1_x)
        # layout of level-1 modules that feed into level-2 module's receptive field
        self.rf1_layout_size = (rf1_layout_y, rf1_layout_x)
        # the extent to which each level-1 module is offset from its neighobors
        # overlap between modules = rf1_{x,y} - rf1_offset_{x,y}
        self.rf1_offset_x = rf1_offset_x
        self.rf1_offset_y = rf1_offset_y
        # receptive field dimensions of the level-2 module (there's only one level-2 module)
        self.rf2_size = (rf1_y+(rf1_layout_y-1)*rf1_offset_y,
                         rf1_x+(rf1_layout_x-1)*rf1_offset_x)
        
        # examine and assign proper values for the image filter
        # DoG: Difference of Gaussians edge detection
        self.DoG_ksize = DoG_ksize
        self.DoG_sigma1 = DoG_sigma1
        self.DoG_sigma2 = DoG_sigma2
        if image_filter == "DoG":
            self.image_filter = image_filter
        # rescale image values to the range provided as a tuple
        elif (type(image_filter) == tuple):
            if (len(image_filter) == 2) & (all([type(x) in [int, float] for x in image_filter])):
                self.image_filter = image_filter
            else:
                self.image_filter = None
        else:
            self.image_filter = None

        # load raw images, filter/rescale each image as a whole
        # divide each image into level-2 patches, multiply their values by a constant (=scale)
        self.load_images(scale, data_dir, image_filter)

        # only shuffle images, not level-2 patches within each image
        if shuffle:
            indices = np.random.permutation(len(self.rf2_patches))
            self.rf2_patches = self.rf2_patches[indices]
            self.labels = self.labels[indices]
        
        # gaussian windowing mask to be applied on each level-1 module
        self.use_mask = use_mask
        self.gauss_mask_sigma = gauss_mask_sigma
        self.mask = self.create_gauss_mask(sigma=gauss_mask_sigma,
                                           width=rf1_x,
                                           height=rf1_y)

    # load all PNG images from the default or user-specified directory
    def load_images(self, scale, data_dir, image_filter):
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
            image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            images.append(image)

        images = np.array(images)
        self.load_sub(images, scale, image_filter)

    def create_gauss_mask(self, sigma=0.4, width=16, height=16):
        """ Create gaussian mask. """
        mu = 0.0
        x, y = np.meshgrid(np.linspace(-1,1,width), np.linspace(-1,1,height))
        d = np.sqrt(x**2+y**2)
        g = np.exp(-( (d-mu)**2 / (2.0*sigma**2) )) / np.sqrt(2.0*np.pi*sigma**2)
        mask = g / np.max(g)
        return mask

    def load_sub(self, images, scale, image_filter):
        """
        Load raw images and filter/rescale each image as a whole.
        Divide each image into level-2 patches and multiply their values by a constant (=scale).
        The resulting matrices have the following dimensions:
        self.images: (number of images, image height, image width)
        self.filtered_images: (number of images, image height, image width)
        self.rf2_patches: (number of images, rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
        self.labels: (number of images, rf2_layout_y, rf2_layout_x, number of images)
        """
        rf2_y, rf2_x = self.rf2_size
        rf2_offset_y, rf2_offset_x = self.rf2_size # no overlap between rf2 patches

        self.images = images
        
        filtered_images = []
        for image in images:
            if self.image_filter == None:
                filtered_image = image
            elif self.image_filter == "DoG":
                filtered_image = self.apply_DoG_filter(image, ksize=self.DoG_ksize, sigma1=self.DoG_sigma1, sigma2=self.DoG_sigma2)
            else:
                filtered_image = np.interp(image, (0, 255), image_filter)
            filtered_images.append(filtered_image)

        self.filtered_images = np.array(filtered_images)

        h, w = images.shape[1:]

        rf2_layout_y, rf2_layout_x = np.array((h, w)) // np.array((rf2_y, rf2_x))

        rf2_patches = np.empty((len(images), rf2_layout_y , rf2_layout_x, rf2_y, rf2_x))
        # different patches of the same training image will be assigned the same identity
        labels = np.empty((len(images), rf2_layout_y , rf2_layout_x, len(images)))

        for image_index, filtered_image in enumerate(filtered_images):
            filtered_image = np.ascontiguousarray(filtered_image)
            shape = [rf2_layout_y, rf2_layout_x, rf2_y, rf2_x]

            patch_strides = filtered_image.itemsize * np.array([w, 1])
            layout_strides = np.array([rf2_offset_y, rf2_offset_x]) * patch_strides
            strides = np.concatenate((layout_strides, patch_strides), axis=None)

            rf2_patch = np.lib.stride_tricks.as_strided(filtered_image, shape=shape, strides=strides)
            rf2_patches[image_index] = rf2_patch
            labels[image_index] = np.tile(np.identity(len(images))[image_index], (rf2_layout_y, rf2_layout_x, 1))

        self.rf2_patches = rf2_patches * scale
        self.labels = labels

    # rf2_patch is 2-dimensional: (rf2_y, rf2_x)
    # the resulting rf1_patches is 4-dimensional: (rf1_layout_y, rf1_layout_x, rf1_y, rf1_x)
    def get_rf1_patches_from_rf2_patch(self, rf2_patch):
        rf1_x = self.rf1_size[1]
        rf1_y = self.rf1_size[0]
        rf1_offset_x = self.rf1_offset_x
        rf1_offset_y = self.rf1_offset_y
        rf1_layout_x = self.rf1_layout_size[1]
        rf1_layout_y = self.rf1_layout_size[0]

        rf2_patch = np.ascontiguousarray(rf2_patch)
        rf2_y, rf2_x = rf2_patch.shape
        shape = [rf1_layout_y, rf1_layout_x, rf1_y, rf1_x]

        patch_strides = rf2_patch.itemsize * np.array([rf2_x, 1])
        layout_strides = np.array([rf1_offset_y, rf1_offset_x]) * patch_strides
        strides = np.concatenate((layout_strides, patch_strides), axis=None)

        rf1_patches = np.lib.stride_tricks.as_strided(rf2_patch, shape=shape, strides=strides)
        if self.use_mask:
            rf1_patches * self.mask

        return rf1_patches
    
    # rf2_patch_index is 3-dimensional: (images, rf2_layout_y, rf2_layout_x)
    def get_rf1_patches(self, rf2_patch_index):
        rf2_patch = self.rf2_patches[rf2_patch_index]
        return self.get_rf1_patches_from_rf2_patch(rf2_patch)

    def apply_DoG_filter(self, image, ksize=(5,5), sigma1=1.3, sigma2=2.6):
        """
        Apply difference of gaussian (DoG) filter detect edge of the image.
        """
        g1 = cv2.GaussianBlur(image, ksize, sigma1)
        g2 = cv2.GaussianBlur(image, ksize, sigma2)
        return g1 - g2

    # NOTE: ignore the following two functions for now
    # def get_bar_images(self, is_short):
    #     patch = self.get_bar_patch(is_short)
    #     return self.get_images_from_patch(patch, use_mask=True)

    # def get_bar_patch(self, is_short):
    #     """
    #     Get bar patch image for end stopping test.
    #     """
    #     bar_patch = np.ones((16,26))
    
    #     if is_short:
    #         bar_width = 6
    #     else:
    #         bar_width = 24
    #     bar_height = 2
    
    #     for x in range(bar_patch.shape[1]):
    #         for y in range(bar_patch.shape[0]):
    #             if x >= 26/2 - bar_width/2 and \
    #             x < 26/2 + bar_width/2 and \
    #             y >= 16/2 - bar_height/2 and \
    #             y < 16/2 + bar_height/2:
    #                 bar_patch[y,x] = -1.0

    #     # Sete scale with stddev of all patch images.
    #     scale = np.std(self.patches)
    #     # Original scaling value for bar
    #     bar_scale = 2.0
    #     return bar_patch * scale * bar_scale
