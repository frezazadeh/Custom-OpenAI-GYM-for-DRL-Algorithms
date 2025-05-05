# Multi-Agent Graph-ESN Reinforcement Learning

A multi-agent reinforcement learning framework for a custom OpenAI Gym environment (`BS-v0`), combining Echo State Networks (ESN) for temporal encoding and Graph Convolutional Networks (GCN) for relational state representation, trained via the REINFORCE algorithm.

## 🚀 Features

- **Custom Gym Environment**  
  - `BS-v0`: CartPole-style base-station simulator  
  - Registered under `gym.make('BS-v0')` with Gym’s new step API  
- **Graph-Reinforce Agent**  
  - **Echo State Network** for lightweight temporal memory  
  - **GCNConv + LayerNorm + global pooling** to capture pairwise state relations  
  - **Policy-gradient (REINFORCE)** with normalized, discounted returns  
- **Multi-Agent Trainer**  
  - Spin up *N* independent agents in parallel, each on its own `BS-v0` instance  
  - On-policy training loop with per-episode ESN reset and policy updates  

## 📦 Core dependencies

pip install gym pygame torch-geometric numpy

