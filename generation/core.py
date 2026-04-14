"""Core dataset transformation logic for PROVGEN and the internal LDP baseline."""

from __future__ import annotations

import numpy as np


def encode(matrix: np.ndarray) -> np.ndarray:
    """Encode ternary SNP values into the binary representation used by PROVGEN."""
    encoded = np.zeros((matrix.shape[0], matrix.shape[1] * 2), dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            if matrix[i, j] == 0:
                encoded[i, 2 * j : 2 * j + 2] = [0, 0]
            elif matrix[i, j] == 1:
                encoded[i, 2 * j : 2 * j + 2] = [0, 1]
            else:
                encoded[i, 2 * j : 2 * j + 2] = [1, 1]
    return encoded


def decode(matrix: np.ndarray) -> np.ndarray:
    """Map the binary representation back into ternary SNP values."""
    decoded = np.zeros((matrix.shape[0], matrix.shape[1] // 2), dtype=int)
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1] // 2):
            decoded[i, j] = matrix[i, 2 * j] + matrix[i, 2 * j + 1]
    return decoded


def get_mafs(encoded_matrix: np.ndarray) -> np.ndarray:
    """Compute minor allele frequencies from a binary-encoded matrix."""
    n, p = encoded_matrix.shape
    return np.sum(encoded_matrix.reshape(n, p // 2, 2), axis=(0, 2)) / np.full(p // 2, n * 2)


def transport(noisy_encoded_matrix: np.ndarray, target_mafs: np.ndarray) -> np.ndarray:
    """Adjust noisy encoded SNPs so their MAFs align with the desired targets."""
    noisy_mafs = get_mafs(noisy_encoded_matrix)
    points_to_flip = ((noisy_mafs - target_mafs) * noisy_encoded_matrix.shape[0] * 2).astype(int)

    for j in range(len(noisy_mafs)):
        if points_to_flip[j] == 0:
            continue
        snp_values = noisy_encoded_matrix[:, 2 * j : 2 * j + 2]
        value_flipped = 1 if points_to_flip[j] > 0 else 0
        value_indices = np.argwhere(snp_values == value_flipped)
        if value_indices.shape[0] == 0:
            continue
        choose_n = min(abs(points_to_flip[j]), value_indices.shape[0])
        chosen = np.random.choice(value_indices.shape[0], choose_n, replace=False)
        indices_to_flip = value_indices[chosen]
        for i, k in indices_to_flip:
            snp_values[i, k] = 1 - value_flipped
        noisy_encoded_matrix[:, 2 * j : 2 * j + 2] = snp_values
    return noisy_encoded_matrix


def xor_mechanism(matrix: np.ndarray, epsilon: float, reference_matrix: np.ndarray) -> np.ndarray:
    """Apply the XOR-based perturbation mechanism described in the paper."""
    eps = np.finfo(float).eps
    matrix = np.array(matrix)
    nrows, ncols = matrix.shape
    sens = ncols

    reference = np.array(reference_matrix)
    ref_nrows, ref_ncols = reference.shape
    m_11 = reference.T @ reference
    ones = np.ones((ref_nrows, ref_ncols))
    m_01 = ones.T @ reference - m_11
    m_10 = reference.T @ ones - m_11
    m_00 = ref_nrows * np.ones((ref_ncols, ref_ncols)) - m_01 - m_11 - m_10
    m_1 = reference.sum(axis=0)
    m_0 = ref_nrows - m_1

    theta_tilde = np.log((m_11 * m_00 + eps) / (m_10 * m_01 + eps))
    diag_val = np.log((m_1 + eps) / (m_0 + eps))
    theta_tilde = theta_tilde - np.diag(np.diag(theta_tilde)) + np.diag(diag_val)

    f_theta = np.linalg.norm(theta_tilde, "fro")
    theta = (epsilon / (sens * f_theta)) * theta_tilde

    off_diagonal = theta - np.diag(np.diag(theta))
    min_value = 2 * np.sum((off_diagonal < 0) * off_diagonal, axis=1) + np.diag(theta)
    max_value = 2 * np.sum((off_diagonal > 0) * off_diagonal, axis=1) + np.diag(theta)
    coeff = np.exp(min_value) - 1
    bound = (coeff < 0) * min_value + (coeff > 0) * max_value
    probabilities = (1 + coeff / (1 + np.exp(bound))) / 2

    flips = np.zeros((nrows, ncols), dtype=int)
    for j in range(nrows):
        flips[j] = np.random.binomial(n=1, p=probabilities, size=ncols)
    return np.logical_xor(matrix, flips).astype(int)


def generate_ldp_dataset(matrix: np.ndarray, epsilon_per_snp: float) -> np.ndarray:
    """Generate the local-DP baseline used in the paper."""
    nrows, ncols = matrix.shape
    keep_probability = np.exp(epsilon_per_snp / ncols) / (np.exp(epsilon_per_snp / ncols) + 2)
    perturbed = np.copy(matrix)
    flip_mask = np.random.binomial(1, keep_probability, size=matrix.shape)
    random_values = np.random.choice([0, 1, 2], size=matrix.shape)
    perturbed[flip_mask == 1] = random_values[flip_mask == 1]
    return perturbed


def generate_proposed_dataset(data: np.ndarray, reference: np.ndarray, epsilon: float) -> np.ndarray:
    """Run the full PROVGEN transformation from original data to released data."""
    encoded = encode(data)
    target_mafs = get_mafs(encoded)
    xor_binary = xor_mechanism(encoded, epsilon, encode(reference))
    return decode(transport(xor_binary, target_mafs))
