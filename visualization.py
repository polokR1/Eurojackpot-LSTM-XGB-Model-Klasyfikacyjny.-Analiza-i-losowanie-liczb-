# visualization.py
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

def plot_frequency(data, root):
    numbers = pd.concat([data['L1'], data['L2'], data['L3'], data['L4'], data['L5']])
    freq = numbers.value_counts().sort_index()
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(freq.index, freq.values, color='skyblue')
    ax.set_title('Częstotliwość występowania liczb głównych')
    ax.set_xlabel('Liczby')
    ax.set_ylabel('Częstotliwość')

    canvas = FigureCanvasTkAgg(fig, master=root)
    canvas.draw()
    canvas.get_tk_widget().pack(pady=10)
