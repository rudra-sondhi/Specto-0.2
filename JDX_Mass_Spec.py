import matplotlib.pyplot as plt
import numpy as np
from jcamp import jcamp_read
import os

from Helper_Functions import read_jdx


def extract_npoints_and_title(data):
    npoints = data.get('NPOINTS', len(data['x']))
    title = data.get('title', 'No Title')
    return npoints, title

def extract_xy_pairs(data):
    x_vals = data.get('x')
    y_vals = data.get('y')
    if x_vals is None or y_vals is None:
        return None
    xy_pairs = [(x, y / 100) for x, y in zip(x_vals, y_vals)]  # Divide y by 100
    return xy_pairs

def plot_xy_pairs(xy_pairs, title, cas_id, save_dir):
    x_vals = [x for x, _ in xy_pairs]
    y_vals = [y for _, y in xy_pairs]

    plt.figure(figsize=(19, 6))
    plt.bar(x_vals, y_vals, color='#d55e00', width=0.6)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.title(title)

    # Add x value on top of each bar with padding, removing decimals
    padding = 2  # Padding value
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + padding, f'{int(x)}', ha='center', va='bottom', fontsize=6)

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f'{cas_id}.png')
    plt.savefig(save_path)

    plt.close()
    print(f"Saved Mass Spec for {cas_id} at {save_path}")
    return save_path


def plot_xy_pairs_no_axes(xy_pairs, cas_id, save_dir):
    x_vals = [x for x, _ in xy_pairs]
    y_vals = [y for _, y in xy_pairs]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x_vals, y_vals, color='#d55e00', width=0.6)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)


    # Add text annotations
    padding = 2  # Padding value
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + padding, f'{int(x)}', ha='center', va='bottom', fontsize=6)

    # Ensure the directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save the plot
    save_path = os.path.join(save_dir, f'{cas_id}.png')
    plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    print(f"Saved NIST-style plot for {cas_id} at {save_path}")

    return save_path



