import matplotlib.pyplot as plt
import numpy as np
from jcamp import jcamp_read

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

def plot_xy_pairs(xy_pairs, title):
    x_vals = [x for x, _ in xy_pairs]
    y_vals = [y for _, y in xy_pairs]

    plt.figure(figsize=(18, 5))
    plt.bar(x_vals, y_vals, color='#d55e00', width=0.6)
    plt.xlabel('m/z')
    plt.ylabel('Relative Intensity')
    plt.title(title)

    # Add x value on top of each bar with padding, removing decimals
    padding = 2  # Padding value
    for x, y in zip(x_vals, y_vals):
        plt.text(x, y + padding, f'{int(x)}', ha='center', va='bottom', fontsize=6)

    plt.show()

'''
def plot_xy_pairs_no_axes(xy_pairs):
    x_vals = [x for x, _ in xy_pairs]
    y_vals = [y for _, y in xy_pairs]

    fig, ax = plt.subplots(figsize=(18, 5))
    ax.bar(x_vals, y_vals, color='#d55e00', width=0.6)
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)

    # Identify the 3 largest and 3 smallest x values
    sorted_x_pairs = sorted(xy_pairs, key=lambda pair: pair[0])
    smallest_x_pairs = sorted_x_pairs[:3]
    largest_x_pairs = sorted_x_pairs[-3:]

    # Identify the 5 largest y values
    sorted_y_pairs = sorted(xy_pairs, key=lambda pair: pair[1], reverse=True)
    largest_y_pairs = sorted_y_pairs[:5]

    # Add text annotations
    padding = 2  # Padding value
    for x, y in smallest_x_pairs + largest_x_pairs + largest_y_pairs:
        plt.text(x, y + padding, f'{int(x)}', ha='center', va='bottom', fontsize=6)

    plt.show()
'''


def plot_xy_pairs_no_axes(xy_pairs):
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

    plt.show()


# Example usage
filename = '/Users/rudrasondhi/Desktop/Specto/Specto/Data/IR_Spectra/Super Cool Data/mass/109-57-9.jdx'
content = read_jdx(filename)

if content:
    npoints, title = extract_npoints_and_title(content)
    xy_pairs = extract_xy_pairs(content)
    
    if xy_pairs is None:
        print("Error: x or y data not found in the content.")
    else:
        if len(xy_pairs) != npoints:
            print(f"Error: Number of extracted points ({len(xy_pairs)}) does not match NPOINTS ({npoints})")
        else:
            plot_xy_pairs(xy_pairs, title)
            plot_xy_pairs_no_axes(xy_pairs)
else:
    print("Error: Failed to read content from the file.")
