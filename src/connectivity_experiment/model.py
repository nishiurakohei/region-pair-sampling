from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    clipped = np.clip(x, -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-clipped))


def binary_cross_entropy_with_logits(logits: np.ndarray, targets: np.ndarray) -> tuple[float, np.ndarray]:
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    losses = np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))
    probs = sigmoid(logits)
    gradient = ((probs - targets) / len(targets)).reshape(-1, 1)
    return float(np.mean(losses)), gradient


def per_sample_binary_loss(logits: np.ndarray, targets: np.ndarray) -> np.ndarray:
    logits = logits.reshape(-1)
    targets = targets.reshape(-1)
    return np.maximum(logits, 0.0) - logits * targets + np.log1p(np.exp(-np.abs(logits)))


def _get_im2col_indices(x_shape, field_height, field_width, stride):
    _, channels, height, width = x_shape
    out_height = (height - field_height) // stride + 1
    out_width = (width - field_width) // stride + 1

    i0 = np.repeat(np.arange(field_height), field_width)
    i0 = np.tile(i0, channels)
    i1 = stride * np.repeat(np.arange(out_height), out_width)
    j0 = np.tile(np.arange(field_width), field_height * channels)
    j1 = stride * np.tile(np.arange(out_width), out_height)
    i = i0.reshape(-1, 1) + i1.reshape(1, -1)
    j = j0.reshape(-1, 1) + j1.reshape(1, -1)
    k = np.repeat(np.arange(channels), field_height * field_width).reshape(-1, 1)
    return k, i, j


def _im2col(x, field_height, field_width, stride):
    k, i, j = _get_im2col_indices(x.shape, field_height, field_width, stride)
    cols = x[:, k, i, j]
    channels = x.shape[1]
    cols = cols.transpose(1, 2, 0).reshape(field_height * field_width * channels, -1)
    return cols


def _col2im(cols, x_shape, field_height, field_width, stride):
    output = np.zeros(x_shape, dtype=cols.dtype)
    k, i, j = _get_im2col_indices(x_shape, field_height, field_width, stride)
    cols_reshaped = cols.reshape(field_height * field_width * x_shape[1], -1, x_shape[0]).transpose(2, 0, 1)
    np.add.at(output, (slice(None), k, i, j), cols_reshaped)
    return output


@dataclass
class Conv2D:
    in_channels: int
    out_channels: int
    kernel_size: int
    seed: int
    stride: int = 1

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / (self.in_channels * self.kernel_size * self.kernel_size))
        self.weights = rng.normal(0.0, scale, size=(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)).astype(np.float32)
        self.biases = np.zeros((self.out_channels, 1), dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)
        self.x_shape = None
        self.x_cols = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.x_shape = x.shape
        self.x_cols = _im2col(x, self.kernel_size, self.kernel_size, self.stride)
        weights_cols = self.weights.reshape(self.out_channels, -1)
        out = weights_cols @ self.x_cols + self.biases
        out_height = (x.shape[2] - self.kernel_size) // self.stride + 1
        out_width = (x.shape[3] - self.kernel_size) // self.stride + 1
        return out.reshape(self.out_channels, out_height, out_width, x.shape[0]).transpose(3, 0, 1, 2)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad_reshaped = grad_output.transpose(1, 2, 3, 0).reshape(self.out_channels, -1)
        self.grad_weights = (grad_reshaped @ self.x_cols.T).reshape(self.weights.shape)
        self.grad_biases = np.sum(grad_reshaped, axis=1, keepdims=True)
        weights_cols = self.weights.reshape(self.out_channels, -1)
        grad_input_cols = weights_cols.T @ grad_reshaped
        return _col2im(grad_input_cols, self.x_shape, self.kernel_size, self.kernel_size, self.stride)


class ReLU:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.mask = x > 0
        return x * self.mask

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output * self.mask


@dataclass
class MaxPool2D:
    kernel_size: int = 2
    stride: int = 2

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        batch_size, channels, height, width = x.shape
        out_height = (height - self.kernel_size) // self.stride + 1
        out_width = (width - self.kernel_size) // self.stride + 1
        output = np.zeros((batch_size, channels, out_height, out_width), dtype=x.dtype)
        self.max_positions = np.zeros((batch_size, channels, out_height, out_width, 2), dtype=np.int32)

        for out_y in range(out_height):
            for out_x in range(out_width):
                region = x[
                    :,
                    :,
                    out_y * self.stride : out_y * self.stride + self.kernel_size,
                    out_x * self.stride : out_x * self.stride + self.kernel_size,
                ]
                flattened = region.reshape(batch_size, channels, -1)
                max_index = np.argmax(flattened, axis=2)
                output[:, :, out_y, out_x] = np.take_along_axis(flattened, max_index[:, :, None], axis=2).squeeze(axis=2)
                self.max_positions[:, :, out_y, out_x, 0] = max_index // self.kernel_size
                self.max_positions[:, :, out_y, out_x, 1] = max_index % self.kernel_size
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        batch_size, channels, out_height, out_width = grad_output.shape
        grad_input = np.zeros_like(self.input)
        for out_y in range(out_height):
            for out_x in range(out_width):
                for batch_index in range(batch_size):
                    for channel_index in range(channels):
                        rel_y, rel_x = self.max_positions[batch_index, channel_index, out_y, out_x]
                        grad_input[
                            batch_index,
                            channel_index,
                            out_y * self.stride + rel_y,
                            out_x * self.stride + rel_x,
                        ] += grad_output[batch_index, channel_index, out_y, out_x]
        return grad_input


class Flatten:
    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        return grad_output.reshape(self.input_shape)


@dataclass
class Linear:
    in_features: int
    out_features: int
    seed: int

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = np.sqrt(2.0 / self.in_features)
        self.weights = rng.normal(0.0, scale, size=(self.in_features, self.out_features)).astype(np.float32)
        self.biases = np.zeros((1, self.out_features), dtype=np.float32)
        self.grad_weights = np.zeros_like(self.weights)
        self.grad_biases = np.zeros_like(self.biases)

    def forward(self, x: np.ndarray) -> np.ndarray:
        self.input = x
        return x @ self.weights + self.biases

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        self.grad_weights = self.input.T @ grad_output
        self.grad_biases = np.sum(grad_output, axis=0, keepdims=True)
        return grad_output @ self.weights.T


class SmallCNN:
    def __init__(self, input_size: int, seed: int = 42, input_channels: int = 3) -> None:
        conv_output_size = ((input_size - 5 + 1) // 2 - 3 + 1) // 2
        flattened_size = conv_output_size * conv_output_size * 8
        self.layers = [
            Conv2D(input_channels, 4, kernel_size=5, seed=seed),
            ReLU(),
            MaxPool2D(),
            Conv2D(4, 8, kernel_size=3, seed=seed + 1),
            ReLU(),
            MaxPool2D(),
            Flatten(),
            Linear(flattened_size, 32, seed=seed + 2),
            ReLU(),
            Linear(32, 1, seed=seed + 3),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights -= learning_rate * (layer.grad_weights + weight_decay * layer.weights)
                layer.biases -= learning_rate * layer.grad_biases

    def predict_logits(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        logits = []
        for start in range(0, len(x), batch_size):
            logits.append(self.forward(x[start : start + batch_size]))
        return np.vstack(logits)

    def save(self, path: str) -> None:
        payload = {}
        for index, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                payload[f"layer_{index}_weights"] = layer.weights
                payload[f"layer_{index}_biases"] = layer.biases
        np.savez(path, **payload)

    def load(self, path: str) -> None:
        payload = np.load(path)
        for index, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layer.weights = payload[f"layer_{index}_weights"]
                layer.biases = payload[f"layer_{index}_biases"]


class LargeReceptiveFieldCNN:
    def __init__(self, input_size: int, seed: int = 42, input_channels: int = 3) -> None:
        size_after_conv1 = input_size - 9 + 1
        size_after_pool1 = size_after_conv1 // 2
        size_after_conv2 = size_after_pool1 - 5 + 1
        size_after_pool2 = size_after_conv2 // 2
        size_after_conv3 = size_after_pool2 - 3 + 1
        flattened_size = size_after_conv3 * size_after_conv3 * 16
        self.layers = [
            Conv2D(input_channels, 8, kernel_size=9, seed=seed),
            ReLU(),
            MaxPool2D(),
            Conv2D(8, 16, kernel_size=5, seed=seed + 1),
            ReLU(),
            MaxPool2D(),
            Conv2D(16, 16, kernel_size=3, seed=seed + 2),
            ReLU(),
            Flatten(),
            Linear(flattened_size, 32, seed=seed + 3),
            ReLU(),
            Linear(32, 1, seed=seed + 4),
        ]

    def forward(self, x: np.ndarray) -> np.ndarray:
        output = x
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad

    def step(self, learning_rate: float, weight_decay: float = 0.0) -> None:
        for layer in self.layers:
            if hasattr(layer, "weights"):
                layer.weights -= learning_rate * (layer.grad_weights + weight_decay * layer.weights)
                layer.biases -= learning_rate * layer.grad_biases

    def predict_logits(self, x: np.ndarray, batch_size: int = 32) -> np.ndarray:
        logits = []
        for start in range(0, len(x), batch_size):
            logits.append(self.forward(x[start : start + batch_size]))
        return np.vstack(logits)

    def save(self, path: str) -> None:
        payload = {}
        for index, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                payload[f"layer_{index}_weights"] = layer.weights
                payload[f"layer_{index}_biases"] = layer.biases
        np.savez(path, **payload)

    def load(self, path: str) -> None:
        payload = np.load(path)
        for index, layer in enumerate(self.layers):
            if hasattr(layer, "weights"):
                layer.weights = payload[f"layer_{index}_weights"]
                layer.biases = payload[f"layer_{index}_biases"]
