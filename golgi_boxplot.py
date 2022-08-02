import click
import os
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd


def load_file(filename):
    x = []
    with open(filename) as f:
        f.readline()
        for line in f:
            name, count, area, avg_area, p, mean = line.split(',')
            if not count == 'NaN':
                x.append(float(count))
    return x





@click.command()
@click.argument('data', nargs=-1)
def main(data):
    fig1, ax1 = plt.subplots()
    series = []
    for d in data:
        data = load_file(d)
        series.append(data)
        # sns.distplot(df, hist=True, kde=True, label=d.split('/')[-1])
    # plt.boxplot(series, showfliers=False)
    # plt.title('are')
    # plt.show()
    # option 1, specify props dictionaries
    c = "orange"
    box2 = plt.boxplot([series[0], series[2]], positions=[1, 3], notch=True, patch_artist=True,
                boxprops=dict(facecolor=c, color=c),
                capprops=dict(color=c),
                whiskerprops=dict(color=c),
                flierprops=dict(color=c, markeredgecolor=c),
                medianprops=dict(color=c), showfliers=False
                )

    # option 2, set all colors individually
    c2 = "purple"
    box1 = plt.boxplot([series[1], series[3]], positions=[2, 4], notch=True, patch_artist=True
                       , showfliers=False)
    for item in ['boxes', 'whiskers', 'fliers', 'medians', 'caps']:
        plt.setp(box1[item], color=c2)
    plt.setp(box1["boxes"], facecolor=c2)
    plt.setp(box1["fliers"], markeredgecolor=c2)

    # plt.xlim(0.5,4)
    ticks = [1, 2, 3, 4]
    labels = ['HR0', 'HR1', 'SR0', 'SR1']
    plt.xticks(ticks, labels)
    plt.legend([box2["boxes"][0], box1["boxes"][0]], ['DMSO', 'Nocodazole'])
    plt.show()


if __name__ == '__main__':
    main()
