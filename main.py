import pygame
import random
import threading
from math import (cos, sin, pi)
from time import sleep


render_queue: list = []
running: bool = True


def render(screen: pygame.Surface, clock: pygame.time.Clock):
    while running:
        screen.fill("black")

        for object in render_queue:
            pygame.draw.rect(screen, pygame.Color(255, 255, 255), object)

        pygame.display.flip()

        clock.tick(999)


def main():
    global running
    global render_queue
    # overall game settings
    pygame.init()
    screen_width, screen_height = (1280, 720)
    screen = pygame.display.set_mode((screen_width, screen_height))
    clock = pygame.time.Clock()

    # player paddle set up
    player_paddle = pygame.Rect(0, 0, 25, 100)
    player_speed = 50

    # ball set up
    ball = pygame.Rect(0, 0, 25, 25).move(pygame.Vector2(screen.get_width() / 2,
                                                         screen.get_height() / 2))
    ball_velocity: int = 5
    ball_angle: float = random.uniform(0.25 * pi, 0.75 * pi)

    render_thread = threading.Thread(target=render, args=(screen, clock,))
    render_thread.start()
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        pre_render_queue: list = []

        # region player logic
        pre_render_queue.append(player_paddle)
        move_player: tuple[int, int] = (0, 0)
        paddle_top, paddle_bottom, paddle_right = (player_paddle.top, player_paddle.bottom,
                                                   player_paddle.right)

        keys = pygame.key.get_pressed()
        if keys[pygame.K_w]:
            if paddle_top - player_speed >= 0:
                move_player = (0, -player_speed)
            else:
                move_player = (0, -paddle_top)
        if keys[pygame.K_s]:
            if paddle_bottom + player_speed + 100 <= screen_height:
                move_player = (0, player_speed)
            else:
                move_player = (0, screen_height - paddle_bottom)

        player_paddle = player_paddle.move(move_player)
        # endregion end player logic

        # region ball logic
        pre_render_queue.append(ball)
        ball_offset: list = []
        ball_y_offset, ball_x_offset = (ball_velocity * cos(ball_angle),
                                        ball_velocity * sin(ball_angle))

        ball_top, ball_bottom, ball_left, ball_right = ball.top, ball.bottom, ball.left, ball.right

        if (ball_top + ball_y_offset >= paddle_top and
            ball_bottom + ball_y_offset <= paddle_bottom and
                ball_left + ball_x_offset <= paddle_right):
            ball_offset = [paddle_right - ball_left, ball_y_offset]
            ball_angle = 2 * pi - ball_angle
        else:
            if ball_right + ball_x_offset <= screen_width and ball_left + ball_x_offset >= 0:
                ball_offset.append(ball_x_offset)
            else:
                if ball_x_offset < 0:
                    ball_offset.append(-ball_left)
                    # ball_angle = 2 * pi - ball_angle
                    running = False
                else:
                    ball_offset.append(screen_width - ball_right)
                    ball_angle = 2 * pi - ball_angle

            if ball_bottom + ball_y_offset <= screen_height and ball_top + ball_y_offset >= 0:
                ball_offset.append(ball_y_offset)
            else:
                if ball_y_offset < 0:
                    ball_offset.append(-ball_top)
                    ball_angle = pi - ball_angle
                else:
                    ball_offset.append(screen_height - ball_bottom)
                    ball_angle = pi - ball_angle

        ball = ball.move(tuple(ball_offset))
        # endregion end ball logic

        render_queue = pre_render_queue

        sleep(1/60)


if __name__ == '__main__':
    main()
