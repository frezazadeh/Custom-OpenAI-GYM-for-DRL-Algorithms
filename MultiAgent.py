import gym
import torch
from Agent import GraphReinforceAgent, Utils, GraphDataProcessor

class MultiAgent:
    def __init__(
        self,
        n_agents: int = 2,
        env_id: str = 'BS-v0',
        device: torch.device = None,
    ):
        import custom_BS

        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.envs = [
            gym.make(env_id, new_step_api=True, disable_env_checker=True)
            for _ in range(n_agents)
        ]
        obs_dim    = self.envs[0].observation_space.shape[0]
        action_dim = self.envs[0].action_space.n
        self.index_pairs = GraphDataProcessor.construct_index_pairs(obs_dim)
        self.agents = [
            GraphReinforceAgent(
                input_dim=obs_dim,
                action_dim=action_dim
            ).to(self.device)
            for _ in range(n_agents)
        ]

    def train(self, episodes: int = 100, gamma: float = 0.99):
        for idx, (env, agent) in enumerate(zip(self.envs, self.agents)):
            print(f"\n=== Training agent #{idx} ===")
            for ep in range(1, episodes + 1):
                agent.esn.state.zero_()
                state, _ = env.reset()
                done = False
                total_reward = 0.0

                while not done:
                    action, log_prob = agent.select_action(state, self.index_pairs)
                    next_state, reward, term, trunc, _ = env.step(action)
                    done = term or trunc
                    agent.store_transition(reward, log_prob)
                    state = next_state
                    total_reward += reward

                agent.optimize(gamma)

                if ep == 1 or ep % 10 == 0:
                    print(f"Agent {idx} | Episode {ep:3d} | Reward: {total_reward:.2f}")

            print(f"--- Agent #{idx} training complete ---")

        for env in self.envs:
            env.close()
