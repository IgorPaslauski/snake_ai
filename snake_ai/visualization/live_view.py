from __future__ import annotations
from dataclasses import dataclass
from typing import List
import pygame
import numpy as np

from snake_ai.env.snake_env import SnakeEnv, SnakeConfig
from snake_ai.agents.neural_net import NeuralNetwork


@dataclass
class LiveViewConfig:
    cell_size: int = 20
    fps: int = 10
    show_grid: bool = True


class LiveViewer:
    """Mostra um agente jogando Snake em tempo real via pygame."""

    def __init__(
        self,
        env_config: SnakeConfig,
        net_arch: List[int],
        params: np.ndarray,
        view_cfg: LiveViewConfig | None = None,
    ):
        self.env = SnakeEnv(env_config)
        self.net = NeuralNetwork(net_arch)
        self.net.set_params(params)
        self.view_cfg = view_cfg or LiveViewConfig()
        self.clock = None
        self.screen = None
        self.paused = False

    def _init_pygame(self):
        pygame.init()
        width = self.env.grid_width * self.view_cfg.cell_size
        height = self.env.grid_height * self.view_cfg.cell_size
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Snake AI - Live View")
        self.clock = pygame.time.Clock()

    def run(self, episodes: int = 1, speed_multiplier: float = 1.0):
        self._init_pygame()
        for _ in range(episodes):
            state = self.env.reset()
            self.paused = False
            done = False

            while not done:
                for event in pygame.event.get():
                    if event.type == pygame.QUIT:
                        pygame.quit()
                        return
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            pygame.quit()
                            return
                        if event.key == pygame.K_SPACE:
                            self.paused = not self.paused

                if self.paused:
                    self.clock.tick(5)
                    continue

                logits = self.net.forward(state)
                action = int(np.argmax(logits))
                next_state, reward, done, info = self.env.step(action)
                state = next_state

                self._draw()
                self.clock.tick(int(self.view_cfg.fps * speed_multiplier))

        pygame.quit()

    def _draw(self):
        cell = self.view_cfg.cell_size
        self.screen.fill((0, 0, 0))

        if self.view_cfg.show_grid:
            for x in range(self.env.grid_width):
                pygame.draw.line(
                    self.screen,
                    (40, 40, 40),
                    (x * cell, 0),
                    (x * cell, self.env.grid_height * cell),
                    1,
                )
            for y in range(self.env.grid_height):
                pygame.draw.line(
                    self.screen,
                    (40, 40, 40),
                    (0, y * cell),
                    (self.env.grid_width * cell, y * cell),
                    1,
                )

        fx, fy = self.env.food
        pygame.draw.rect(
            self.screen,
            (200, 30, 30),
            pygame.Rect(fx * cell, fy * cell, cell, cell),
        )

        for i, (x, y) in enumerate(self.env.snake):
            color = (0, 200, 0) if i == 0 else (0, 120, 0)
            pygame.draw.rect(
                self.screen,
                color,
                pygame.Rect(x * cell, y * cell, cell, cell),
            )

        pygame.display.flip()
