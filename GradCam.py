
import torch
import cv2

# Define the GradCAM class
class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        # Register hooks to save activations and gradients
        self.target_layer.register_forward_hook(self.save_activation)
        self.target_layer.register_backward_hook(self.save_gradient)

    def save_activation(self, module, input, output):
        self.activations = output.detach()

    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_image, class_idx=None):
        self.model.zero_grad()
        output = self.model(input_image)
        # If no class index is specified, choose the predicted class
        if class_idx is None:
            class_idx = torch.argmax(output, dim=1)
        # Compute loss for the target class
        loss = output[:, class_idx].sum()
        loss.backward()

        # Compute the weights by global-average pooling the gradients
        weights = torch.mean(self.gradients, dim=[2, 3], keepdim=True)
        cam = torch.sum(weights * self.activations, dim=1)
        cam = torch.relu(cam)
        cam = cam.squeeze().cpu().numpy()
        # Resize to the original image size (assumes input_image shape is [1, C, H, W])
        cam = cv2.resize(cam, (input_image.shape[3], input_image.shape[2]))
        # Normalize between 0 and 1
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam