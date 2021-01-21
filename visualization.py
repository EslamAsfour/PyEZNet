from matplotlib import pyplot as plt
import pandas as pd
import matplotlib.animation as animation
from matplotlib import style
import subprocess
import sys

def draw_error (epoches, loss):
    plt.plot(epoches, loss)
    plt.xlabel("epoches")
    plt.ylabel("loss")
    plt.show()

def draw_error_animation (): 
    style.use('fivethirtyeight')
    fig = plt.figure()
    global ax1 
    ax1 = fig.add_subplot(1,1,1)
    ani = animation.FuncAnimation(fig, animate, interval=1000)
    plt.show()

def animate(i):
    graph_data = open('example.txt','r+').read()
    lines = graph_data.split('\n')
    xs = []
    ys = []
    for line in lines:
        if len(line) > 1:
            x, y = line.split(',')
            xs.append(float(x))
            ys.append(float(y))
    ax1.clear()
    ax1.plot(xs, ys)

subprocess.Popen(["python", "test_mlp.py"] + sys.argv[1:])
draw_error_animation()