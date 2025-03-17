
import torch
import cv2

# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.hooks = []
        self.gradients = None
        self.activations = None

        # Register hooks
        self.hooks.append(target_layer.register_forward_hook(self.save_activation))
        self.hooks.append(target_layer.register_full_backward_hook(self.save_gradient))

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):
        # Forward pass
        model_output = self.model(input_tensor)

        if class_idx is None:
            class_idx = torch.argmax(model_output)

        # Zero gradients
        self.model.zero_grad()

        # Target for backprop
        one_hot = torch.zeros_like(model_output)
        one_hot[0, class_idx] = 1

        # Backward pass
        model_output.backward(gradient=one_hot, retain_graph=True)

        # Get weights from gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)

        # Weight activation maps
        cam = torch.sum(weights * self.activations, dim=1).squeeze()

        # ReLU on CAM
        cam = torch.maximum(cam, torch.tensor(0.0))

        # Normalize
        if torch.max(cam) > 0:
            cam = cam / torch.max(cam)

        # Resize to input size
        cam = cam.cpu().numpy()
        cam = cv2.resize(cam, (224, 224))
        return cam

    def __del__(self):
        for hook in self.hooks:
            hook.remove()
