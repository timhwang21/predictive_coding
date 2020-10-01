# -*- coding: utf-8 -*-
import numpy as np
import os


class Model:
    def __init__(self, dataset):
        self.dataset = dataset
        self.iteration = 30
        self.prior = "kurtotic" # "kurtotic" or "gaussian"
        
        # NOTE: k_r (k1) and k_U (k2) do not match with that from the original paper
        self.k_r = 0.0005 # Learning rate for r
        self.k_U_init = 0.005  # Initial learning rate for U
        self.k_U = self.k_U_init

        self.sigma_sq0 = 1.0  # Variance of observation distribution of I
        self.sigma_sq1 = 10.0 # Variance of observation distribution of r1
        self.sigma_sq2 = 10.0 # Variance of observation distribution of r2
        self.sigma_sq3 = 2.0 # Variance of observation distribution of r3
        self.alpha1 = 1.0  # Precision param of r1 prior
        self.alpha2 = 0.05 # Precision param of r2 prior
        self.alpha3 = 0.05 # Precision param of r3 prior
        
        # NOTE: the original paper only provides one lambda value (which is lambda1 here)
        self.lambda1 = 0.02 # Precision param of U1 prior
        self.lambda2 = 0.00001 # Precision param of U2 prior
        self.lambda3 = 0.00001 # Precision param of U3 prior
        
        # Level 1 consists of multiple modules with 32 neurons in each moodule
        # Level-1 modules' receptive fields can be arranged into a 1D or 2D grid, with overlap between neighboring receptive fields
        self.level1_x = dataset.rf1_size[1]
        self.level1_y = dataset.rf1_size[0]
        self.level1_offset_x = dataset.rf1_offset_x
        self.level1_offset_y = dataset.rf1_offset_y
        self.level1_layout_x = dataset.rf1_layout_size[1]
        self.level1_layout_y = dataset.rf1_layout_size[0]
        self.level1_module_size = 32

        # Level 2 consists of one module with 128 neurons whose receptive field covers that of all level-1 modules
        self.level2_module_size = 128

        # Level 3 consists of localist nodes, one for each training image for classification
        self.level3_module_size = len(dataset.images)

        # Top-down weights
        self.U1 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_y, self.level1_x, self.level1_module_size) - 0.5
        self.U2 = np.random.rand(self.level1_layout_y, self.level1_layout_x, self.level1_module_size, self.level2_module_size) - 0.5
        self.U3 = np.random.rand(self.level2_module_size, self.level3_module_size) - 0.5

    def prior_trans(self, x, prior):
        if prior == "kurtotic":
            x_trans = 1 + np.square(x)
        else:
            x_trans = 1
        return x_trans

    # inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
    # labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
    def apply_input(self, inputs, labels, dataset, training):
        outputs = {'index':[],
                   'input': [],
                   'label': [],
                   'iteration': [],
                   'r1': [],
                   'r2': [],
                   'r3': [],
                   'e1': [],
                   'e2': [],
                   'e3': []}

        # state estimates (r hats)
        inputs = np.array(inputs)
        r1 = np.zeros((self.level1_layout_y, self.level1_layout_x, self.level1_module_size), dtype=np.float32)
        r2 = np.zeros(self.level2_module_size, dtype=np.float32)
        r3 = np.zeros(self.level3_module_size, dtype=np.float32)
    
        for idx in np.ndindex(inputs.shape[:2]):
            inputs_idx = dataset.get_rf1_patches_from_rf2_patch(inputs[idx])
            labels_idx = labels[idx]

            for i in range(self.iteration):
                # state predictions (r bars)
                r10 = np.array([self.U1[i, j] @ r1[i, j] for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(inputs_idx.shape)
                r21 = self.U2.dot(r2).reshape(r1.shape)
                r32 = self.U3.dot(r3)
                r43 = labels_idx if training else labels_idx*0

                # prediction errors
                e0 = inputs_idx - r10
                e1 = r1 - r21
                e2 = r2 - r32
                e3 = (np.exp(r3)/np.sum(np.exp(r3))) - r43 # softmax cross-entropy loss

                # r updates
                dr1 = (self.k_r/self.sigma_sq0) * np.array([np.tensordot(self.U1[i, j], e0[i, j], axes=((0, 1), (0, 1))) for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(r1.shape) \
                    + (self.k_r/self.sigma_sq1) * -e1 \
                    - self.k_r * self.alpha1 * r1 / self.prior_trans(r1, self.prior)

                dr2 = (self.k_r / self.sigma_sq1) * np.tensordot(self.U2, e1, axes=((0, 1, 2),(0, 1, 2))) \
                    + (self.k_r / self.sigma_sq2) * -e2 \
                    - self.k_r * self.alpha2 * r2 / self.prior_trans(r2, self.prior)

                dr3 = (self.k_r / self.sigma_sq2) * self.U3.T.dot(e2) \
                    + (self.k_r / self.sigma_sq3) * -e3 \
                    - self.k_r * self.alpha3 * r3 / self.prior_trans(r3, self.prior)

                # U updates
                if training:
                    dU1 = (self.k_U/self.sigma_sq0) * np.array([e0[i, j, :, :, None] @ r1[i, j, None, None, :] for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x)]).reshape(self.U1.shape) \
                        - self.k_U * self.lambda1 * self.U1 / self.prior_trans(self.U1, self.prior)

                    dU2 = (self.k_U / self.sigma_sq1) * e1[:, :, :, None] @ r2[None, None, None, :] \
                        - self.k_U * self.lambda2 * self.U2 / self.prior_trans(self.U2, self.prior)

                    dU3 = (self.k_U / self.sigma_sq2) * e2[:, None] @ r3[None, :] \
                        - self.k_U * self.lambda3 * self.U3 / self.prior_trans(self.U3, self.prior)

                # apply r updates
                r1 += dr1
                r2 += dr2
                r3 += dr3

                # apply U updates
                if training:
                    self.U1 += dU1
                    self.U2 += dU2
                    self.U3 += dU3
                
                # append outputs
                outputs['index'].append(idx)
                outputs['input'].append(inputs[idx])
                outputs['label'].append(labels[idx])
                outputs['iteration'].append(i)
                outputs['r1'].append(r1.copy())
                outputs['r2'].append(r2.copy())
                outputs['r3'].append(r3.copy())
                outputs['e1'].append(e1.copy())
                outputs['e2'].append(e2.copy())
                outputs['e3'].append(e3.copy())

        return outputs

    # training on rf2_patches in order within a given image
    def train(self, dataset):
        # images are presented in the order defined in dataset
        # rf2 patches of a given image are presented in an ascending sequence (moves through x before y)
        # inputs = rf2_patches is 4D: (rf2_layout_y, rf2_layout_x, rf2_y, rf2_x)
        # labels is 3D: (rf2_layout_y, rf2_layout_x, number of images)
        for i in range(dataset.rf2_patches.shape[0]):
            inputs = dataset.rf2_patches[i]
            labels = dataset.labels[i]
            outputs = self.apply_input(inputs, labels, dataset, training=True)

        print("train finished")

    # representations at each level
    # values change with input
    def reconstruct(self, r, level=1):
        if level == 1:
            r1 = r
        elif level == 2:
            r2 = r
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
        elif level == 3:
            r3 = r
            r2 = self.U3.dot(r3)
            r1 = self.U2.dot(r2).reshape((self.level1_layout_y, self.level1_layout_x, self.level1_module_size))
            
        rf2_patch = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                              self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)
        
        # reconstruct each of the three rf1 patches separately and then combine
        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            r = r1[i, j]
            U = self.U1[i, j]
            Ur = U.dot(r)
            rf2_patch[self.level1_offset_y * i : self.level1_offset_y * i + self.level1_y, \
                      self.level1_offset_x * j : self.level1_offset_x * j + self.level1_x] += Ur
        return rf2_patch

    # rf: receptive field
    # values don't change with input (when model is not being trained)
    def get_level2_rf(self, index):
        rf = np.zeros((self.level1_y + (self.level1_offset_y * (self.level1_layout_y - 1)), \
                       self.level1_x + (self.level1_offset_x * (self.level1_layout_x - 1))), dtype=np.float32)

        for i, j in np.ndindex(self.level1_layout_y, self.level1_layout_x):
            U2 = self.U2[i, j, :, index]
            UU = self.U1[i, j].dot(U2)
            rf[self.level1_offset_y * i :self.level1_offset_y * i + self.level1_y, \
               self.level1_offset_x * j :self.level1_offset_x * j + self.level1_x] += UU

        return rf

    def save(self, dir_name):
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
        file_path = os.path.join(dir_name, "model") 

        np.savez_compressed(file_path,
                            U1=self.U1,
                            U2=self.U2,
                            U3=self.U3)
        print("saved: {}".format(dir_name))

    def load(self, dir_name):
        file_path = os.path.join(dir_name, "model.npz")
        if not os.path.exists(file_path):
            print("saved file not found")
            return
        data = np.load(file_path)
        self.U1 = data["U1"]
        self.U2 = data["U2"]
        self.U3 = data["U3"]
        print("loaded: {}".format(dir_name))
