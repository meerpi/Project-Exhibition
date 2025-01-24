# mediapipe hand skeleton as a graph for ST-GCN
# 21 landmarks per hand, 42 total (left 0-20, right 21-41)

import numpy as np
import torch

# edges for a single hand - each (i, j) is a bone connection
# fmt: off
SINGLE_HAND_EDGES = [
    (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
    (0, 5), (5, 6), (6, 7), (7, 8),       # index
    (0, 9), (9, 10), (10, 11), (11, 12),   # middle
    (0, 13), (13, 14), (14, 15), (15, 16), # ring
    (0, 17), (17, 18), (18, 19), (19, 20), # pinky
    (5, 9), (9, 13), (13, 17),             # palm cross
    # (5, 17),  # tried adding thumb-pinky edge, made accuracy worse
]
# fmt: on

NUM_JOINTS_PER_HAND = 21


def build_hand_adjacency(num_hands=2, add_cross_hand=True):
    """build raw adjacency matrix for hand skeleton graph"""
    num_nodes = NUM_JOINTS_PER_HAND * num_hands
    A = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    for hand_idx in range(num_hands):
        offset = hand_idx * NUM_JOINTS_PER_HAND
        for i, j in SINGLE_HAND_EDGES:
            A[offset + i, offset + j] = 1.0
            A[offset + j, offset + i] = 1.0

    # connect wrists across hands
    if num_hands == 2 and add_cross_hand:
        A[0, NUM_JOINTS_PER_HAND] = 1.0
        A[NUM_JOINTS_PER_HAND, 0] = 1.0

    return A


def normalize_adjacency(A):
    """symmetric norm: A_hat = D^-1/2 (A+I) D^-1/2"""
    A_hat = A + np.eye(A.shape[0], dtype=np.float32)
    D_hat = np.diag(A_hat.sum(axis=1))
    D_inv_sqrt = np.diag(1.0 / np.sqrt(D_hat.diagonal() + 1e-8))
    return D_inv_sqrt @ A_hat @ D_inv_sqrt


def get_adjacency_matrix(num_hands=2, add_cross_hand=True):
    """normalized adjacency as pytorch tensor"""
    A = build_hand_adjacency(num_hands, add_cross_hand)
    A_norm = normalize_adjacency(A)
    return torch.from_numpy(A_norm)


if __name__ == "__main__":
    A_raw = build_hand_adjacency(num_hands=2, add_cross_hand=True)
    A_norm = get_adjacency_matrix(num_hands=2, add_cross_hand=True)

    num_edges = int(A_raw.sum()) // 2
    print(A_raw.shape)  # quick check
    print(f"Graph: {A_raw.shape[0]} nodes, {num_edges} edges")
    print(f"Normalized adj: shape={A_norm.shape}, symmetric={torch.allclose(A_norm, A_norm.T)}")
    print(f"Row sums: min={A_norm.sum(1).min():.3f}, max={A_norm.sum(1).max():.3f}")
