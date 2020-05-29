# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, dataset, iteration=30):
        self.dataset = dataset
        self.iteration = iteration
        
        # NOTE: k1 and k2 do not match with that from the original paper
        self.k1      = 0.0005 # Learning rate for r
        self.k2_init = 0.005  # Initial learning rate for U
        # k2 decreases gradually by dividing with 1.015 every 40 training inputs
        self.k2_decay_cycle = 40
        self.k2_decay_rate = 1.015

        self.sigma_sq    = 1.0  # Variance of observation distribution of I
        self.sigma_sq_td = 10.0 # Variance of observation distribution of r
        self.sigma_sq_3 = 10.0
        self.sigma_sq_4 = 10.0
        self.alpha1      = 1.0  # Precision param of r prior    (var=1.0,  std=1.0)
        self.alpha2      = 0.05 # Precision param of r_td prior (var=20.0, std=4.5)
        self.alpha3 = 0.05
        
        # NOTE: the original paper only provides one lambda value (which is lambd1 here)
        self.lambd1      = 0.02 # Precision param of U prior    (var=50.0, std=7.1)
        self.lambd2      = 0.00001 # Precision param of Uh prior
        self.lambd3 = 0.00001
        
        # NOTE: NOT SURE WHAT THIS SCALE IS FOR
        # U: weight matrix
        U_scale = 1.0
        
        # inputs: three 16 x 16 overlapping image patches (offset by 5 pixels horizontally)
        # Level 1: 3 modules, each module has 32 input-estimating neurons and 32 error-detecting neurons
        # Level 2: 128 input-estimating neurons and 128 error-detecting neurons
        self.input_x = dataset.rf1_size[1]
        self.input_y = dataset.rf1_size[0]
        self.input_size = self.input_x * self.input_y
        self.input_offset_x = dataset.rf1_offset_x
        self.input_offset_y = dataset.rf1_offset_y

        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_n = self.level1_layout_x * self.level1_layout_y
        self.level1_module_size = 32
        
        self.level2_module_size = 128

        # Us: level-1 top-down weights
        # 3 modules (one for each image patch); 256 pixels (16 x 16) in each image patch; 32 input-estimating neurons in each module
        self.Us = (np.random.rand(self.level1_module_n, self.input_size, self.level1_module_size) - 0.5) * U_scale
        # Uh: level-2 top-down weights
        # 96 (3 x 32) level-1 error-detecting neurons; 128 level-2 input-estimating neurons
        self.Uh = (np.random.rand(self.level1_module_n * self.level1_module_size, self.level2_module_size) - 0.5) * U_scale

        self.k2 = self.k2_init

        # NOTE: NOT SURE WHAT THIS SCALE IS FOR
        # Scaling parameter for learning rate of level2
        self.level2_lr_scale = 1.0

        # level 3 for classification (level-3 consists of localisi nodes, one for each training image)
        self.level3_module_size = len(dataset.images)
        self.U3 = (np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5) * U_scale

    def apply_images(self, images, label, training):
        # 96 (3 x 32) level-1 input-estimating neurons' representations
        rs = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)
        # 128 level-2 input-estimating neurons' representations
        rh = np.zeros([self.level2_module_size], dtype=np.float32)
        # level-3 representations
        r3 = np.zeros([self.level3_module_size], dtype=np.float32)
        
        # 96 (3 x 32) level-1 within-level prediction error
        error_tds = np.zeros([self.level1_module_n * self.level1_module_size], dtype=np.float32)
    
        for i in range(self.iteration):
            # Loop for iterations

            # Calculate r_td (level 2's prediction for level 1)
            r_tds = self.Uh.dot(rh) # (96,)

            for j in range(self.level1_module_n):
                # corresponding neuron range from level1 vector
                v_start = self.level1_module_size * j
                v_end = self.level1_module_size * (j+1)
                # input
                I = images[j]
                # level-1 input estimates
                r = rs[v_start:v_end]
                # level 2's predictions for level 1
                r_td = r_tds[v_start:v_end]

                # weights from level 1 to level 0 (input)
                U  = self.Us[j] # (256,32)
                # level 1's predictions for input
                Ur = U.dot(r) # (256,)

                # level 1's bottom-up prediction error (from input to level 1)
                error    = I - Ur # (256,)
                # level 1's within-level prediction error (based on level 2's predictions)
                error_td = r_td - r # (32,)
                
                # gradient descent on E (optimization function) with respect to r, assuming Gaussian prior distribution
                # Equation 7
                # NOTE: seems to assume the activation function is linear here (f(x) = x) instead of tanh(x)
                dr = (self.k1/self.sigma_sq) * U.T.dot(error) \
                     + (self.k1/self.sigma_sq_td) * error_td \
                     - self.k1 * self.alpha1 * r
                     # (32,)

                # gradient descent on E (optimization function) with respect to U
                # Equation 9
                if training:
                    dU = (self.k2/self.sigma_sq) * np.outer(error, r) \
                         - self.k2 * self.lambd1 * U
                         # (256,32)
                    self.Us[j] += dU

                rs[v_start:v_end] += dr
                error_tds[v_start:v_end] = error_td

            # Level2 update
            error_3 = rh - self.U3.dot(r3) # (128,)
            error_4 = r3 - label

            # gradient descent on E (optimization function) with respect to r, assuming Gaussian prior distribution
            # Equation 7
            # -error_tds = r - r_td: level 2's bottom-up prediction error (from level 1 to level 2)
            drh = (self.k1*self.level2_lr_scale / self.sigma_sq_td) * self.Uh.T.dot(-error_tds) \
                  + (self.k1*self.level2_lr_scale / self.sigma_sq_3) * -error_3 \
                  - self.k1*self.level2_lr_scale * self.alpha2 * rh
                  # (128,)
            
            # gradient descent on E (optimization function) with respect to U
            # Equation 9
            if training:
                dUh = (self.k2*self.level2_lr_scale / self.sigma_sq_td) * np.outer(-error_tds, rh) \
                      - self.k2*self.level2_lr_scale * self.lambd2 * self.Uh
                # (96,128)
                self.Uh += dUh

            rh += drh

            # level 3 (classification) update
            dr3 = (self.k1 / self.sigma_sq_3) * self.U3.T.dot(error_3) \
                + (self.k1 / self.sigma_sq_4) * -error_4 \
                  - self.k1 * self.alpha3 * r3
            
            if training:
                dU3 = (self.k2 / self.sigma_sq_3) * np.outer(error_3, r3) \
                      - self.k2 * self.lambd3 * self.U3
                
                self.U3 += dU3
            
            r3 += dr3
            r3 = np.exp(r3)/sum(np.exp(r3)) # softmax

        return rs, r_tds, rh, error_tds, r3

    def train(self, dataset):
        self.k2 = self.k2_init
        self.labels = dataset.labels
        
        patch_size = len(dataset.patches) # 2375

        for i in range(patch_size):
            # Loop for all patches
            images = dataset.get_images(i)
            label = self.labels[i]
            rs, r_tds, rh, error_tds, r3 = self.apply_images(images, label, training=True)
            
            if i % 100 == 0:
                print("rs    std={:.2f}".format(np.std(rs)))
                print("r_tds std={:.2f}".format(np.std(r_tds)))
                print("U     std={:.2f}".format(np.std(self.Us)))
                print("Uh    std={:.2f}".format(np.std(self.Uh)))
    
            if i % self.k2_decay_cycle == 0:
                # Decay learning rate for U
                self.k2 = self.k2 / self.k2_decay_rate

        print("train finished")

    # representations at each level
    # values change with input
    def reconstruct(self, r, level=1):
        if level==1:
            rs = r # (96,)
        else:
            rh = r # (128,)
            rs = self.Uh.dot(rh) # (96,)
            
        # reconstructed image size is 16 x 26 because the each set of inputs is three overlapping (offset by 5 pixels horizontally) 16 x 16 image patches
        patch = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                          self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)
        
        # reconstruct each of the three patches separately and then combine
        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            r = rs[self.level1_module_size * i:self.level1_module_size * (i+1)]
            U = self.Us[i]
            Ur = U.dot(r).reshape(self.input_y, self.input_x)
            patch[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
                  self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += Ur
        return patch

    # rf: receptive field
    # values don't change with input (when model is not being trained)
    def get_level2_rf(self, index):
        rf = np.zeros((self.input_y + (self.input_offset_y * (self.level1_layout_y - 1)), \
                       self.input_x + (self.input_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)

        for i in range(self.level1_module_n):
            module_y = i % self.level1_layout_y
            module_x = i // self.level1_layout_y

            Uh = self.Uh[:,index][self.level1_module_size * i:self.level1_module_size * (i+1)]
            UU = self.Us[i].dot(Uh).reshape((self.input_y, self.input_x))
            rf[self.input_offset_y * module_y :self.input_offset_y * module_y + self.input_y, \
               self.input_offset_x * module_x :self.input_offset_x * module_x + self.input_x] += UU

        return rf

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            Us=self.Us,
                            Uh=self.Uh)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        data = np.load(file_path)
        self.Us = data["Us"]
        self.Uh = data["Uh"]
        print("loaded: {}".format(dir_name))
