import pygame
import random
import os
import sys
import pickle
import numpy as np

pygame.init()

# --- Settings ---
WIDTH, HEIGHT = 600, 400
CELL_SIZE = 20
ROWS, COLS = HEIGHT // CELL_SIZE, WIDTH // CELL_SIZE

WHITE = (255, 255, 255)
GREEN = (0, 255, 0)
DARK_GREEN = (0, 200, 0)
RED = (255, 0, 0)
BLACK = (0, 0, 0)
GRAY = (40, 40, 40)

screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Snake Game")

font = pygame.font.SysFont("comicsans", 28)
title_font = pygame.font.SysFont("comicsans", 48, bold=True)
clock = pygame.time.Clock()
HS_FILE = "highscore.txt"

# --- Q-Learning Agent ---
class QLearningAgent:
    def __init__(self, lr=0.1, gamma=0.9, epsilon=0.0):
        self.lr = lr
        self.gamma = gamma
        self.epsilon = epsilon
        self.Q = {}

    def get_action(self, state):
        if state not in self.Q:
            self.Q[state] = [0, 0, 0, 0]
        if np.random.rand() < self.epsilon:
            return random.choice([0,1,2,3])
        return int(np.argmax(self.Q[state]))

# --- Helper functions ---
def load_highscore():
    if os.path.exists(HS_FILE):
        with open(HS_FILE, "r") as f:
            return int(f.read().strip())
    return 0

def save_highscore(score):
    high = load_highscore()
    if score > high:
        with open(HS_FILE, "w") as f:
            f.write(str(score))

def draw_text(text, font, color, x, y, center=True):
    render = font.render(text, True, color)
    rect = render.get_rect(center=(x, y)) if center else render.get_rect(topleft=(x, y))
    screen.blit(render, rect)

def draw_grid():
    for x in range(0, WIDTH, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (x, 0), (x, HEIGHT))
    for y in range(0, HEIGHT, CELL_SIZE):
        pygame.draw.line(screen, GRAY, (0, y), (WIDTH, y))

def draw_snake(snake):
    for segment in snake:
        pygame.draw.rect(screen, DARK_GREEN, (*segment, CELL_SIZE, CELL_SIZE))

def draw_food(food):
    pygame.draw.rect(screen, RED, (*food, CELL_SIZE, CELL_SIZE))

# --- Relative state for Q-learning ---
def get_state(snake, food, direction):
    head_x, head_y = snake[0]
    food_dx = (food[0] - head_x) // CELL_SIZE
    food_dy = (food[1] - head_y) // CELL_SIZE

    dx, dy = 0, 0
    if direction == "UP": dx, dy = 0, -1
    elif direction == "DOWN": dx, dy = 0, 1
    elif direction == "LEFT": dx, dy = -1, 0
    elif direction == "RIGHT": dx, dy = 1, 0

    danger_front = int([head_x + dx, head_y + dy] in snake or 
                       head_x + dx < 0 or head_x + dx >= WIDTH or 
                       head_y + dy < 0 or head_y + dy >= HEIGHT)
    danger_left = int([head_x - dy, head_y + dx] in snake or 
                      head_x - dy < 0 or head_x - dy >= WIDTH or 
                      head_y + dx < 0 or head_y + dx >= HEIGHT)
    danger_right = int([head_x + dy, head_y - dx] in snake or 
                       head_x + dy < 0 or head_x + dy >= WIDTH or 
                       head_y - dx < 0 or head_y - dx >= HEIGHT)

    return (danger_front, danger_left, danger_right, 
            int(food_dx < 0), int(food_dx > 0),
            int(food_dy < 0), int(food_dy > 0),
            int(direction=="UP"), int(direction=="DOWN"),
            int(direction=="LEFT"), int(direction=="RIGHT"))

# --- Game loops ---
def game_loop():
    snake = [[100, 100]]
    direction = "RIGHT"
    food = [random.randrange(0, WIDTH, CELL_SIZE),
            random.randrange(0, HEIGHT, CELL_SIZE)]
    score = 0

    while True:
        clock.tick(10)
        # Human input
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP and direction != "DOWN": direction = "UP"
                elif event.key == pygame.K_DOWN and direction != "UP": direction = "DOWN"
                elif event.key == pygame.K_LEFT and direction != "RIGHT": direction = "LEFT"
                elif event.key == pygame.K_RIGHT and direction != "LEFT": direction = "RIGHT"

        x, y = snake[0]
        if direction == "UP": y -= CELL_SIZE
        elif direction == "DOWN": y += CELL_SIZE
        elif direction == "LEFT": x -= CELL_SIZE
        elif direction == "RIGHT": x += CELL_SIZE
        new_head = [x, y]

        if (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or new_head in snake):
            save_highscore(score)
            return score

        snake.insert(0, new_head)

        if new_head == food:
            score += 50
            while True:
                food = [random.randrange(0, WIDTH, CELL_SIZE),
                        random.randrange(0, HEIGHT, CELL_SIZE)]
                if food not in snake:
                    break
        else:
            snake.pop()
            score -= 1  # move penalty

        # Draw
        screen.fill(BLACK)
        draw_grid()
        draw_snake(snake)
        draw_food(food)
        draw_text(f"Score: {score}", font, WHITE, 10, 10, center=False)
        pygame.display.flip()

def ai_game_loop():
    agent = QLearningAgent()
    agent.Q = pickle.load(open("qtable.pkl", "rb"))

    snake = [[100, 100]]
    direction = "RIGHT"
    food = [random.randrange(0, WIDTH, CELL_SIZE),
            random.randrange(0, HEIGHT, CELL_SIZE)]
    score = 0

    while True:
        clock.tick(25)
        state = get_state(snake, food, direction)
        action = agent.get_action(state)

        if action == 0 and direction != "DOWN": direction = "UP"
        elif action == 1 and direction != "UP": direction = "DOWN"
        elif action == 2 and direction != "RIGHT": direction = "LEFT"
        elif action == 3 and direction != "LEFT": direction = "RIGHT"

        x, y = snake[0]
        if direction == "UP": y -= CELL_SIZE
        elif direction == "DOWN": y += CELL_SIZE
        elif direction == "LEFT": x -= CELL_SIZE
        elif direction == "RIGHT": x += CELL_SIZE
        new_head = [x, y]

        if (x < 0 or x >= WIDTH or y < 0 or y >= HEIGHT or new_head in snake):
            save_highscore(score)
            return score

        snake.insert(0, new_head)

        if new_head == food:
            score += 50
            while True:
                food = [random.randrange(0, WIDTH, CELL_SIZE),
                        random.randrange(0, HEIGHT, CELL_SIZE)]
                if food not in snake:
                    break
        else:
            snake.pop()
            score -= 1

        # Draw
        screen.fill(BLACK)
        draw_grid()
        draw_snake(snake)
        draw_food(food)
        draw_text(f"AI Score: {score}", font, WHITE, 10, 10, center=False)
        pygame.display.flip()

# --- Start & game over screens ---
def start_screen():
    selected = 0
    options = ["Play", "AI Play", "Quit"]
    highscore = load_highscore()

    while True:
        screen.fill(BLACK)
        draw_text("SNAKE GAME", title_font, GREEN, WIDTH//2, 100)
        draw_text(f"High Score: {highscore}", font, WHITE, WIDTH//2, 150)

        for i, opt in enumerate(options):
            color = GREEN if i == selected else GRAY
            draw_text(opt, font, color, WIDTH//2, 230 + i*40)

        draw_text("Arrow Keys to move | ENTER to select",
                  pygame.font.SysFont("comicsans", 20),
                  WHITE, WIDTH//2, 340)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_UP:
                    selected = (selected - 1) % len(options)
                elif event.key == pygame.K_DOWN:
                    selected = (selected + 1) % len(options)
                elif event.key == pygame.K_RETURN:
                    if options[selected] == "Play":
                        score = game_loop()
                        game_over_screen(score)
                    elif options[selected] == "AI Play":
                        score = ai_game_loop()
                        game_over_screen(score)
                    elif options[selected] == "Quit":
                        pygame.quit(); sys.exit()

def game_over_screen(score):
    high = load_highscore()
    while True:
        screen.fill(BLACK)
        draw_text("Game Over!", title_font, RED, WIDTH//2, 120)
        draw_text(f"Score: {score}", font, WHITE, WIDTH//2, 180)
        draw_text(f"High Score: {high}", font, GREEN, WIDTH//2, 220)
        draw_text("Press ENTER to return to menu",
                  pygame.font.SysFont("comicsans", 20),
                  WHITE, WIDTH//2, 300)
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit(); sys.exit()
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN:
                return

# --- Run ---
if __name__ == "__main__":
    start_screen()
