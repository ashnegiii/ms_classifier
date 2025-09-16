import torch
import torch.nn.functional as F
import numpy as np

class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hooks = []

        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            if grad_output[0] is not None:
                self.gradients = grad_output[0].detach()

        self.hooks.append(target_layer.register_forward_hook(forward_hook))
        self.hooks.append(target_layer.register_full_backward_hook(backward_hook))

    def generate(self, class_idx, outputs, img_tensor):
        # Reset and do forward pass
        self.model.zero_grad()
        _ = self.model(img_tensor)
        
        # Backward pass
        outputs[0, class_idx].backward(retain_graph=True)

        # Get gradients and activations - we know the shapes now:
        # gradients: [1, 1408, 7, 7] -> average to get [1408] weights
        # activations: [1408, 7, 7] -> use directly
        weights = torch.mean(self.gradients[0], dim=(1, 2))  # [1408]
        activations = self.activations[0]  # [1408, 7, 7]
        
        # Weighted combination: each weight[i] * activation_map[i]
        cam = torch.zeros(activations.shape[1:], device=activations.device)  # [7, 7]
        for i in range(len(weights)):
            cam += weights[i] * activations[i]  # scalar * [7, 7]
        
        # ReLU and normalize
        cam = F.relu(cam)
        if cam.max() > 0:
            cam = cam / cam.max()

        # Resize to input image size
        cam = F.interpolate(
            cam.unsqueeze(0).unsqueeze(0),  # [1, 1, 7, 7]
            size=(img_tensor.shape[2], img_tensor.shape[3]),  # [224, 224]
            mode="bilinear",
            align_corners=False,
        )
        
        return cam.squeeze().cpu().numpy()

    def __del__(self):
        for hook in self.hooks:
            hook.remove()