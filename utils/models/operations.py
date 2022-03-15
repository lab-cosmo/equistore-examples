import numpy as np
import scipy
import torch


def einsum(operation, *arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.einsum(operation, *arrays)
    elif isinstance(arrays[0], torch.Tensor):
        return torch.einsum(operation, *arrays)
    else:
        raise ValueError("unknown array type")


def vstack(arrays):
    if isinstance(arrays[0], np.ndarray):
        return np.vstack(arrays)
    elif isinstance(arrays[0], torch.Tensor):
        return torch.vstack(arrays)
    else:
        raise ValueError("unknown array type")


def norm(array, axis=None):
    if isinstance(array, np.ndarray):
        return np.linalg.norm(array, axis=axis)
    elif isinstance(array, torch.Tensor):
        if axis is None:
            return torch.linalg.norm(array)
        else:
            return torch.linalg.norm(array, dim=axis)
    else:
        raise ValueError("unknown array type")


def float_power(array, power):
    if isinstance(array, np.ndarray):
        return np.float_power(array, power)
    elif isinstance(array, torch.Tensor):
        return torch.float_power(array, power)
    else:
        raise ValueError("unknown array type")


def zeros_like(array, shape):
    if isinstance(array, np.ndarray):
        return np.zeros(shape, dtype=array.dtype)
    elif isinstance(array, torch.Tensor):
        return torch.zeros(shape, dtype=array.dtype)
    else:
        raise ValueError("unknown array type")


def sum(array, axis=None):
    if isinstance(array, np.ndarray):
        return np.sum(array, axis=axis)
    elif isinstance(array, torch.Tensor):
        if axis is None:
            return torch.sum(array)
        else:
            return torch.sum(array, dim=axis)
    else:
        raise ValueError("unknown array type")


def sqrt(array):
    if isinstance(array, np.ndarray):
        return np.sqrt(array)
    elif isinstance(array, torch.Tensor):
        return torch.sqrt(array)
    else:
        raise ValueError("unknown array type")


def array_like(array, data):
    if isinstance(array, np.ndarray):
        return np.array(data)
    elif isinstance(array, torch.Tensor):
        return torch.tensor(data)
    else:
        raise ValueError("unknown array type")


def linalg_inv(array, detach):
    if isinstance(array, np.ndarray):
        return np.linalg.inv(array)
    elif isinstance(array, torch.Tensor):
        result = torch.linalg.inv(array)
        if detach:
            result = result.detach()
        return result
    else:
        raise ValueError("unknown array type")


def linalg_solve(A, B, detach):
    if isinstance(A, np.ndarray):
        return np.linalg.lstsq(A, B, rcond=None)[0]
    elif isinstance(A, torch.Tensor):
        result = torch.linalg.solve(A, B)
        if detach:
            result = result.detach()
        return result
    else:
        raise ValueError("unknown array type")


def block_diag(*arrays):
    if isinstance(arrays[0], np.ndarray):
        return scipy.linalg.block_diag(*arrays)
    elif isinstance(arrays[0], torch.Tensor):
        return torch.block_diag(*arrays)
    else:
        raise ValueError("unknown array type")
