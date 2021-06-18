from pytorch_grad_cam.grad_cam import GradCAM
from pytorch_grad_cam.activations_and_gradients import ActivationsAndGradientsMmdet
import cv2
import numpy as np
import torch
import ttach as tta

class MMdetCam(GradCAM):
    def __init__(self, 
                 model, 
                 target_layer,
                 use_cuda=False,
                 reshape_transform=None):
        self.model = model.eval()
        self.target_layer = target_layer
        self.cuda = use_cuda
        if self.cuda:
            self.model = model.cuda()
        self.reshape_transform = reshape_transform
        self.activations_and_grads = ActivationsAndGradientsMmdet(self.model, 
            target_layer, reshape_transform)


    def forward(self, input_tensor, scale_idx=0, target_category=None, eigen_smooth=False):

        if self.cuda:
            input_tensor = input_tensor.cuda()

        output = self.activations_and_grads(input_tensor)

        output = output[scale_idx].permute(0, 2, 3, 1).reshape(1, -1)

        if type(target_category) is int:
            target_category = [target_category] * input_tensor.size(0)

        if target_category is None:
            target_category = np.argmax(output.cpu().data.numpy(), axis=-1)
        else:
            assert(len(target_category) == input_tensor.size(0))

        self.model.zero_grad()
        loss = self.get_loss(output, target_category)
        loss.backward(retain_graph=True)

        activations = self.activations_and_grads.activations[-1].cpu().data.numpy()
        grads = self.activations_and_grads.gradients[-1].cpu().data.numpy()

        cam = self.get_cam_image(input_tensor, target_category, 
            activations, grads, eigen_smooth)

        cam = np.maximum(cam, 0)

        result = []
        for img in cam:
            img = cv2.resize(img, input_tensor.shape[-2:][::-1])
            img = img - np.min(img)
            img = img / np.max(img)
            result.append(img)
        result = np.float32(result)
        return result, loss

    def __call__(self,
                 input_tensor,
                 scale_idx=0,
                 target_category=None,
                 aug_smooth=False,
                 eigen_smooth=False):
        if aug_smooth is True:
            return self.forward_augmentation_smoothing(input_tensor,
                target_category, eigen_smooth)

        return self.forward(input_tensor, scale_idx,
            target_category, eigen_smooth)