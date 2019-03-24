import numpy as np
from math import ceil
import matplotlib.pyplot as plt


def process(file):
    print("="*50)
    print(file)
    print("-"*50)
    losses = []
    scores = []

    with open(file, 'r') as fp:
        segments = [[_.strip() for _ in line.strip().split("|")] for line in fp]

        array0 = [float(segm[2].split(":")[1]) for segm in segments]
        array1 = [float(segm[3].split(":")[1]) for segm in segments]
        array2 = [float(segm[4].split(":")[1]) for segm in segments]
        array3 = [float(segm[5].split(":")[1]) for segm in segments]

        for i in range(ceil(len(array1) / 1000)):
            mean0 = np.mean(np.array(array0[i * 1000: i * 1000 + 1000]))
            mean1 = np.mean(np.array(array1[i * 1000: i * 1000 + 1000]))
            mean2 = np.mean(np.array(array2[i * 1000: i * 1000 + 1000]))
            mean3 = np.mean(np.array(array3[i * 1000: i * 1000 + 1000]))
            losses.append(mean2)
            scores.append(mean3)
            print(f"{i+1:>2d} {mean0:>6.2f} {mean1:>10.3f} {mean2:>5.4f} {mean3:>5.4f}")
    print()
    return losses, scores


def plot_loss(losses):
    plt.figure(figsize=[12, 8])
    plt.title("average loss per 1000 games")
    plt.xlabel("games")
    plt.ylabel("mean squared error")

    plt.plot(range(len(losses)), losses)
    plt.grid(True)
    plt.savefig("loss_coins.svg")


def plot_score(scores):
    plt.figure(figsize=[12, 8])
    plt.title("average score per 1000 games")
    plt.xlabel("games")
    plt.ylabel("score")

    plt.plot(range(len(scores)), scores)
    plt.yticks(range(10))
    plt.grid(True)
    plt.savefig("score_coins.svg")


if __name__ == "__main__":
    ls, ss = process("./statistics/stats_2.txt")

    plot_loss(ls)
    plot_score(ss)
