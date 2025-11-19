from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple
import random
import numpy as np


Direction = Tuple[int, int]  # (dx, dy)


@dataclass
class SnakeConfig:
    grid_width: int = 20
    grid_height: int = 20
    max_steps_without_food: int = 200
    step_penalty: float = -0.001
    food_reward: float = 1.0
    death_penalty: float = -1.0


class SnakeEnv:
    """
    Ambiente do jogo Snake em grid.
    Observação retornada é um vetor np.array com:
    [danger_straight, danger_right, danger_left,
     move_dir_up, move_dir_down, move_dir_left, move_dir_right,
     food_up, food_down, food_left, food_right]
    """

    def __init__(self, config: SnakeConfig):
        self.config = config
        self.grid_width = config.grid_width
        self.grid_height = config.grid_height
        self.reset()

    def reset(self) -> np.ndarray:
        self.direction: Direction = (1, 0)  # começa indo para a direita
        x = self.grid_width // 2
        y = self.grid_height // 2
        self.snake: List[Tuple[int, int]] = [(x, y), (x - 1, y), (x - 2, y)]
        self.score = 0
        self.steps = 0
        self.steps_since_food = 0
        self._place_food()
        self.done = False
        return self._get_state()

    def _place_food(self) -> None:
        while True:
            food = (
                random.randint(0, self.grid_width - 1),
                random.randint(0, self.grid_height - 1),
            )
            if food not in self.snake:
                self.food = food
                break

    def _is_collision(self, position: Tuple[int, int]) -> bool:
        x, y = position
        if x < 0 or x >= self.grid_width or y < 0 or y >= self.grid_height:
            return True
        if position in self.snake[1:]:
            return True
        return False

    def _get_state(self) -> np.ndarray:
        head_x, head_y = self.snake[0]
        dir_x, dir_y = self.direction

        # direções relativas
        left_dir = (-dir_y, dir_x)
        right_dir = (dir_y, -dir_x)

        def danger_in_direction(d: Direction) -> int:
            new_pos = (head_x + d[0], head_y + d[1])
            return int(self._is_collision(new_pos))

        danger_straight = danger_in_direction(self.direction)
        danger_right = danger_in_direction(right_dir)
        danger_left = danger_in_direction(left_dir)

        move_up = int(dir_y == -1)
        move_down = int(dir_y == 1)
        move_left = int(dir_x == -1)
        move_right = int(dir_x == 1)

        food_x, food_y = self.food
        food_up = int(food_y < head_y)
        food_down = int(food_y > head_y)
        food_left = int(food_x < head_x)
        food_right = int(food_x > head_x)

        state = np.array(
            [
                danger_straight,
                danger_right,
                danger_left,
                move_up,
                move_down,
                move_left,
                move_right,
                food_up,
                food_down,
                food_left,
                food_right,
            ],
            dtype=np.float32,
        )
        return state

    def step(self, action: int):
        """
        action: 0 = virar esquerda, 1 = seguir reto, 2 = virar direita
        """
        if self.done:
            raise RuntimeError("Chame reset() antes de continuar.")

        self.steps += 1
        self.steps_since_food += 1

        # muda direção baseada na ação
        dir_x, dir_y = self.direction
        if action == 0:  # esquerda
            self.direction = (-dir_y, dir_x)
        elif action == 2:  # direita
            self.direction = (dir_y, -dir_x)
        # action == 1 mantém direção

        dir_x, dir_y = self.direction
        new_head = (self.snake[0][0] + dir_x, self.snake[0][1] + dir_y)

        reward = self.config.step_penalty

        if self._is_collision(new_head):
            self.done = True
            reward += self.config.death_penalty
            return self._get_state(), float(reward), True, {"score": self.score}

        self.snake.insert(0, new_head)

        if new_head == self.food:
            self.score += 1
            reward += self.config.food_reward
            self.steps_since_food = 0
            self._place_food()
        else:
            self.snake.pop()

        if self.steps_since_food > self.config.max_steps_without_food:
            self.done = True
            reward += self.config.death_penalty

        return self._get_state(), float(reward), self.done, {"score": self.score}
