import torch
from settings import s


class CombinedPerception:

    def __init__(self, use_wall=False):
        self.__use_wall = use_wall

    def __call__(self, environment):
        perception = torch.stack(tuple(self.__create_perception(envir) for envir in environment))
        perception = perception.float().unsqueeze(0)
        return perception

    def __create_perception(self, environment):
        import numpy as np
        arena, me, others, coins, bomb_xys, bomb_map = environment

        combined_view = np.zeros(arena.shape, dtype=int)
        combined_view[arena == -1] = 32  # Walls
        combined_view[arena == 1] = 64  # Crates

        if len(bomb_xys) > 0:
            combined_view[tuple(zip(*bomb_xys))[::-1]] = 96  # Bombs
        if len(coins) > 0:
            combined_view[tuple(zip(*coins))[::-1]] = 128  # Coins

        for bx, by in bomb_xys:
            value = 140 + 10*bomb_map[by, bx]  # explosion 140 - 180
            combined_view[by, bx] = value
            CombinedPerception.__right_explosion(combined_view, by, bx, value, s.cols)
            CombinedPerception.__left_explosion(combined_view, by, bx, value)
            CombinedPerception.__bot_explosion(combined_view, by, bx, value, s.rows)
            CombinedPerception.__top_explosion(combined_view, by, bx, value)

        if len(others) > 0:
            combined_view[tuple(zip(*others))] = 192  # enemies
        combined_view[me[1], me[0]] = 204  # me

        if (me[:2]) in bomb_xys:
            combined_view[me[1], me[0]] = 204 + (10*bomb_map[by, bx] + 1)  # me on bomb (204 no bomb, 254 max)

        perception = torch.from_numpy(combined_view).float() / 255

        if not self.__use_wall:
            perception = perception[1:-1, 1:-1]

        return perception

    @staticmethod
    def __right_explosion(combined_view, by, bx, value, width):
        for dx in range(s.bomb_power):
            if bx + dx + 1 >= width:
                break
            if combined_view[by, bx + (dx + 1)] == 32:
                break
            else:
                combined_view[by, bx + (dx + 1)] = value

    @staticmethod
    def __left_explosion(combined_view, by, bx, value):
        for dx in range(s.bomb_power):
            if bx - (dx + 1) < 0:
                break
            if combined_view[by, bx - (dx + 1)] == 32:
                break
            else:
                combined_view[by, bx - (dx + 1)] = value

    @staticmethod
    def __bot_explosion(combined_view, by, bx, value, height):
        for dy in range(s.bomb_power):
            if by + dy + 1 >= height:
                break
            if combined_view[by + (dy + 1), bx] == 32:
                break
            else:
                combined_view[by + (dy + 1), bx] = value

    @staticmethod
    def __top_explosion(combined_view, by, bx, value):
        for dy in range(s.bomb_power):
            if by - (dy + 1) < 0:
                break
            if combined_view[by - (dy + 1), bx] == 32:
                break
            else:
                combined_view[by - (dy + 1), bx] = value
