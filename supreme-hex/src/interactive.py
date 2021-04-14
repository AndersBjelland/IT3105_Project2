import math
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from .games import Hex


colors = [(0.3, 0.5, 0.4, c) for c in np.linspace(0, 1, 100)]
policy_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    'policy_cmap', colors, N=128)


ANGLE = math.pi / 6
dxx, dyx = math.cos(-ANGLE), math.sin(-ANGLE)
dxy, dyy = math.cos(math.pi+ANGLE), math.sin(math.pi+ANGLE)


def drawing_coords(x, y):
    tx = dxx * x + dxy * y
    ty = dyx * x + dyy * y
    return (tx / 100, ty / 100)


def inverse_drawing_coords(tx, ty):
    x = (-100 * dxy * ty + 100 * dyy * tx) / (dxx * dyy - dxy * dyx)
    y = (100 * dxx * ty - 100 * dyx * tx) / (dxx * dyy - dxy * dyx)

    return round(x), round(y)


def play(game: Hex, agent, heat=lambda edge: edge.N):
    original = game
    plt.ion()
    fig = game._fig(close=False)
    ax = fig.gca()
    heatmap = np.zeros((game.size, game.size))

    XY = np.array([drawing_coords(x, y) for x, y in game.nodes()])
    X, Y = XY[:, 0], XY[:, 1]

    def redraw():
        ax.clear()
        game._fig(ax=ax, close=False)
        fig.canvas.draw()
        fig.canvas.flush_events()

    def on_key_press(event):
        nonlocal game, fig, ax, heat, heatmap
        if event.key == 'escape':
            plt.close()
            return

        if event.key == 'w':
            def heat(edge): return edge.W
            return

        if event.key == 'n':
            def heat(edge): return edge.N
            return

        if event.key == 'p':
            def heat(edge): return edge.P
            return

        if event.key == 'r':
            game = original
            redraw()
            return

        if event.key == 'z':
            heatmap *= 0
            (distribution, z) = agent.prediction(game)
            print(
                f'predicted z = {z} (+1 if player {game.current_player} wins, -1 if the other does)')
            for (x, y) in game.available_actions():
                heatmap[x, y] = distribution[y][x]

            scatter = fig.gca().scatter(X, Y, s=64**2 * (4 / game.size)**2, c=heatmap.reshape((-1,)),
                                        cmap=policy_cmap, edgecolor='none', zorder=200, marker='H')

            fig.canvas.draw()
            fig.canvas.flush_events()

            scatter.remove()
            return

        if event.key == 'd':
            heatmap *= 0
            distribution = agent.policy_distribution(game)

            for (x, y) in game.available_actions():
                heatmap[x, y] = distribution[y][x]

            scatter = fig.gca().scatter(X, Y, s=64**2 * (4 / game.size)**2, c=heatmap.reshape((-1,)),
                                        cmap=policy_cmap, edgecolor='none', zorder=200, marker='H')

            fig.canvas.draw()
            fig.canvas.flush_events()

            scatter.remove()
            return

        if event.key == ' ':
            def callback(node):
                nonlocal heatmap
                heatmap *= 0

                for ((x, y), edge) in node.statistics.items():
                    heatmap[x, y] = heat(edge)

                scatter = fig.gca().scatter(X, Y, s=64**2 * (4 / game.size)**2, c=heatmap.reshape((-1,)),
                                            cmap=policy_cmap, edgecolor='none', zorder=200, marker='H')

                fig.canvas.draw()
                fig.canvas.flush_events()

                scatter.remove()

            action = agent.policy(game, callback=callback)
            game = game.next_state(action)

            redraw()
            return

        print(f'key {repr(event.key)} not recognized')

    def on_click(event):
        nonlocal game
        if event.xdata is None or event.ydata is None:
            return

        (tx, ty) = event.xdata, event.ydata
        x, y = inverse_drawing_coords(tx, ty)
        game = game.next_state((x, y))
        redraw()

    kid = fig.canvas.mpl_connect('key_press_event', on_key_press)
    cid = fig.canvas.mpl_connect('button_press_event', on_click)

    fig.show()
