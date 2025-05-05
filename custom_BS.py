import math
from typing import Optional, Union

import numpy as np
import gym
from gym import logger, spaces
from gym.envs.registration import register
from gym.envs.classic_control import utils
import pygame
from pygame import gfxdraw

# 1) Register under BS-v0, new_step_api so we get a 5-tuple back.
register(
    id='BS-v0',
    entry_point='custom_BS:BSEnv',
    max_episode_steps=500,    # same limit as CartPole-v1
    new_step_api=True,
)

class BSEnv(gym.Env[np.ndarray, Union[int, np.ndarray]]):
    """
    CartPole clone, exposed as 'BS-v0'.
    """

    metadata = {
        "render_modes": ["human", "rgb_array"],
        "render_fps": 50,
    }

    def __init__(self, render_mode: Optional[str] = None):
        # Physics parameters
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = self.masspole + self.masscart
        self.length = 0.5
        self.polemass_length = self.masspole * self.length
        self.force_mag = 10.0
        self.tau = 0.02

        # Termination thresholds
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        # Spaces
        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max,
        ], dtype=np.float32)

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(-high, high, dtype=np.float32)

        # Render
        self.render_mode = render_mode
        self.screen_width = 600
        self.screen_height = 400
        self.screen = None
        self.clock = None

        self.state = None
        self.steps_beyond_terminated = None

    def step(self, action):
        assert self.action_space.contains(action), "invalid action"
        x, x_dot, theta, theta_dot = self.state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta, sintheta = math.cos(theta), math.sin(theta)

        temp = (force + self.polemass_length * theta_dot**2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0/3.0 - self.masspole * costheta**2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler integration
        x       += self.tau * x_dot
        x_dot   += self.tau * xacc
        theta   += self.tau * theta_dot
        theta_dot += self.tau * thetaacc

        self.state = (x, x_dot, theta, theta_dot)

        done = bool(
            x < -self.x_threshold or x > self.x_threshold
            or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        )

        if not done:
            reward = 1.0
        elif self.steps_beyond_terminated is None:
            self.steps_beyond_terminated = 0
            reward = 1.0
        else:
            self.steps_beyond_terminated += 1
            reward = 0.0

        if self.render_mode == "human":
            self.render()

        # new_step_api=True → return 5-tuple
        return np.array(self.state, dtype=np.float32), reward, done, False, {}

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        low, high = utils.maybe_parse_reset_bounds(options, -0.05, 0.05)
        self.state = self.np_random.uniform(low=low, high=high, size=(4,))
        self.steps_beyond_terminated = None
        if self.render_mode == "human":
            self.render()
        # new_step_api=True → return (obs, info)
        return np.array(self.state, dtype=np.float32), {}

    def render(self):
        if self.render_mode is None:
            logger.warn("render() called without render_mode set.")
            return

        if self.screen is None:
            pygame.init()
            if self.render_mode == "human":
                pygame.display.init()
                self.screen = pygame.display.set_mode((self.screen_width, self.screen_height))
            else:
                self.screen = pygame.Surface((self.screen_width, self.screen_height))
        if self.clock is None:
            self.clock = pygame.time.Clock()

        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        polewidth = 10.0
        polelen = scale * (2*self.length)
        cartwidth = 50.0
        cartheight = 30.0

        x, _, theta, _ = self.state
        surf = pygame.Surface((self.screen_width, self.screen_height))
        surf.fill((255,255,255))

        # draw cart
        l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
        cartx = x*scale + self.screen_width/2
        carty = 100
        coords = [(c[0]+cartx, c[1]+carty) for c in [(l,b),(l,t),(r,t),(r,b)]]
        gfxdraw.aapolygon(surf, coords, (0,0,0)); gfxdraw.filled_polygon(surf, coords, (0,0,0))

        # draw pole
        pole_coords = []
        for coord in [(-polewidth/2, -polewidth/2),
                      (-polewidth/2, polelen-polewidth/2),
                      (polewidth/2, polelen-polewidth/2),
                      (polewidth/2, -polewidth/2)]:
            v = pygame.math.Vector2(coord).rotate_rad(-theta)
            pole_coords.append((v.x+cartx, v.y+carty+cartheight/4.0))
        gfxdraw.aapolygon(surf, pole_coords, (202,152,101))
        gfxdraw.filled_polygon(surf, pole_coords, (202,152,101))

        # axle
        gfxdraw.aacircle(surf, int(cartx), int(carty+cartheight/4.0), int(polewidth/2), (129,132,203))
        gfxdraw.filled_circle(surf, int(cartx), int(carty+cartheight/4.0), int(polewidth/2), (129,132,203))
        gfxdraw.hline(surf, 0, self.screen_width, carty, (0,0,0))

        surf = pygame.transform.flip(surf, False, True)
        self.screen.blit(surf, (0,0))
        if self.render_mode == "human":
            pygame.event.pump()
            self.clock.tick(self.metadata["render_fps"])
            pygame.display.flip()
        else:
            return np.transpose(np.array(pygame.surfarray.pixels3d(self.screen)), (1,0,2))

    def close(self):
        if self.screen is not None:
            pygame.display.quit()
            pygame.quit()
            self.screen = None
