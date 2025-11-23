import numpy as np
import random

CELL_SIZE = 20
WIDTH, HEIGHT = 600, 400
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

class SnakeEnv:
    def __init__(self):
        self.reset()

    def reset(self):
        self.direction = "RIGHT"
        self.snake = [[5, 5]]
        self.score = 0

        self._place_food()
        return self._get_state()

    def _place_food(self):
        self.food = [
            random.randint(0, COLS - 1),
            random.randint(0, ROWS - 1)
        ]
        if self.food in self.snake:
            self._place_food()

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food

        # Basic state representation
        return np.array([
            head_x, head_y,
            food_x, food_y,
            int(self.direction == "UP"),
            int(self.direction == "DOWN"),
            int(self.direction == "LEFT"),
            int(self.direction == "RIGHT"),
        ], dtype=np.float32)

    def step(self, action):
        # 0 = up, 1 = down, 2 = left, 3 = right
        if action == 0 and self.direction != "DOWN": self.direction = "UP"
        elif action == 1 and self.direction != "UP": self.direction = "DOWN"
        elif action == 2 and self.direction != "RIGHT": self.direction = "LEFT"
        elif action == 3 and self.direction != "LEFT": self.direction = "RIGHT"

        x, y = self.snake[0]
        if self.direction == "UP":    y -= 1
        if self.direction == "DOWN":  y += 1
        if self.direction == "LEFT":  x -= 1
        if self.direction == "RIGHT": x += 1

        new_head = [x, y]

        # Check collisions
        done = (
            x < 0 or x >= COLS or
            y < 0 or y >= ROWS or
            new_head in self.snake
        )

        reward = -10 if done else 0

        if not done:
            self.snake.insert(0, new_head)

            if new_head == self.food:
                reward = +10
                self.score += 1
                self._place_food()
            else:
                self.snake.pop()

        return self._get_state(), reward, done, {}
