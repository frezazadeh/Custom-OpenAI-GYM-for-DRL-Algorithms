import torch
from Agent import Utils
from MultiAgent import MultiAgent
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

def main():
    # set reproducible seed
    seed = 1234
    Utils.set_seed(seed)

    # choose device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # instantiate multi-agent trainer: 1 agents by default
    ma = MultiAgent(n_agents=1, env_id='BS-v0', device=device)

    # train each agent for 250 episodes with discount factor 0.99
    ma.train(episodes=250, gamma=0.99)

if __name__ == "__main__":
    main()
