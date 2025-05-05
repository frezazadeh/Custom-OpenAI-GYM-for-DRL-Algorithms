import random
from itertools import permutations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv, LayerNorm, global_add_pool
from torch.optim.lr_scheduler import StepLR


class Utils:
    @staticmethod
    def set_seed(seed: int):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.backends.cudnn.enabled:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True


class GraphDataProcessor:
    @staticmethod
    def construct_index_pairs(num_nodes: int):
        index_pairs = [(i, i + 1) for i in range(0, num_nodes - 1, 2)]
        index_pairs += [(i, num_nodes - i - 1) for i in range(num_nodes // 2)]
        return index_pairs

    @staticmethod
    def create_graph_data(state: np.ndarray, index_pairs):
        edge_index = torch.tensor(
            list(permutations(range(len(state)), 2)), dtype=torch.long
        ).t().contiguous()
        node_feats = torch.tensor(
            [[state[i], state[j]] for i, j in index_pairs], dtype=torch.float
        )
        return Data(x=node_feats, edge_index=edge_index)


class EchoStateNetwork(nn.Module):
    def __init__(
        self,
        input_dim: int,
        reservoir_size: int,
        spectral_radius: float = 0.9,
        sparsity: float = 0.5,
        leaky_rate: float = 0.2
    ):
        super().__init__()
        self.reservoir_size = reservoir_size
        self.leaky_rate = leaky_rate
        self.W_in = nn.Parameter(
            (torch.rand(reservoir_size, input_dim) - 0.5) * 2 / input_dim,
            requires_grad=False
        )
        W = torch.rand(reservoir_size, reservoir_size) - 0.5
        mask = torch.rand(reservoir_size, reservoir_size) > sparsity
        W[mask] = 0.0
        v = torch.rand(reservoir_size, 1)
        for _ in range(50):
            v = W @ v
            v = v / v.norm()
        max_eig = v.norm()
        W = W * (spectral_radius / max_eig)
        self.register_buffer("W", W)
        self.register_buffer("state", torch.zeros(reservoir_size))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        prev = self.state
        pre = self.W_in @ x + self.W @ prev
        updated = (1 - self.leaky_rate) * prev + self.leaky_rate * torch.tanh(pre)
        self.state = updated / updated.norm().clamp(min=1e-6)
        return self.state


class GraphReinforceAgent(nn.Module):
    def __init__(
        self,
        input_dim: int,
        action_dim: int,
        esn_reservoir_size: int = 500,
        hidden_dim: int = 128,
        lr: float = 5e-4,
        scheduler_step: int = 100,
        scheduler_gamma: float = 0.1,
    ):
        super().__init__()
        self.esn = EchoStateNetwork(input_dim, esn_reservoir_size)
        self.gcn = GCNConv(2, hidden_dim)
        self.norm = LayerNorm(hidden_dim)
        self.fc1 = nn.Linear(hidden_dim + esn_reservoir_size, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, action_dim)
        self.optimizer = optim.Adam(self.parameters(), lr=lr)
        self.scheduler = StepLR(self.optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        self.memory = []

    def forward(
        self,
        node_feats: torch.Tensor,
        edge_index: torch.Tensor,
        esn_state: torch.Tensor
    ) -> torch.Tensor:
        x = F.relu(self.gcn(node_feats, edge_index))
        batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        x = global_add_pool(self.norm(x), batch)  # shape (1, hidden_dim)
        x = torch.cat([x, esn_state], dim=1)
        x = F.relu(self.fc1(x))
        logits = self.fc2(x)
        return F.log_softmax(logits, dim=1)

    def select_action(self, state: np.ndarray, index_pairs) -> (int, torch.Tensor):
        st = torch.tensor(state, dtype=torch.float, device=self.esn.state.device)
        esn_out = self.esn(st).unsqueeze(0)
        data = GraphDataProcessor.create_graph_data(state, index_pairs)
        log_probs = self.forward(
            data.x.to(esn_out.device),
            data.edge_index.to(esn_out.device),
            esn_out
        )
        dist = Categorical(logits=log_probs)
        action = dist.sample()
        return action.item(), log_probs[0, action]

    def store_transition(self, reward: float, log_prob: torch.Tensor):
        self.memory.append((reward, log_prob))

    def optimize(self, gamma: float):
        returns = []
        R = 0
        for r, _ in reversed(self.memory):
            R = r + gamma * R
            returns.insert(0, R)
        returns = torch.tensor(returns, dtype=torch.float, device=self.memory[0][1].device)
        returns = (returns - returns.mean()) / (returns.std(unbiased=False) + 1e-6)
        loss = 0
        for (r, log_prob), R in zip(self.memory, returns):
            loss = loss - log_prob * R
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.scheduler.step()
        self.memory.clear()
