from PIL import Image
import sys
import numpy as np
from skimage.measure import label
from skimage.color import rgb2gray
from skimage.filters import threshold_niblack
from math import dist
from collections import deque
import argparse
import os

LINE_WIDTH=6
POINT_WIDTH=20

def get_matching_pixels(img, threshold, invert):
    pixels = np.asarray(img)
    threshold = np.array([threshold, threshold, threshold])
    if invert:
        return np.all(threshold < pixels, axis=-1)
    else:
        return np.all(threshold > pixels, axis=-1)

def label_components(pixels):
    pixels = pixels.astype(int)
    labeled_pixels, num_labels = label(pixels, background=0, return_num=True)
    components = [ (np.count_nonzero(labeled_pixels == i), i) for i in range(1, num_labels) ]
    components.sort(reverse=True)
    return components, labeled_pixels

def largest_connected_component(pixels, is_valid_cc):
    components, labeled_pixels = label_components(pixels)
    for size, label in components:
        masked = labeled_pixels == label
        if is_valid_cc(masked):
            return masked
    return None

def mask_o_ring(pixels):
    return largest_connected_component(pixels, lambda x: True)

def get_outermost_pixels(pixels): 
    pixel_coords = np.argwhere(pixels)
    max_y = [None] * len(pixels[0])
    max_x = [None] * len(pixels)
    min_y = [None] * len(pixels[0])
    min_x = [None] * len(pixels)
    for px in np.stack(np.nonzero(pixels), axis=-1):
        x = px[1]
        y = px[0]
        if pixels[y, x]:
            if not min_y[x] or min_y[x] > y:
                min_y[x] = y
            if not max_y[x] or max_y[x] < y:
                max_y[x] = y
            if not min_x[y] or min_x[y] > x:
                min_x[y] = x
            if not max_x[y] or max_x[y] < x:
                max_x[y] = x
    outermost = set()
    for x, y in enumerate(max_y):
        if y:
            outermost.add((x, y))
    for x, y in enumerate(min_y):
        if y:
            outermost.add((x, y))
    for y, x in enumerate(max_x):
        if x:
            outermost.add((x, y))
    for y, x in enumerate(min_x):
        if x:
            outermost.add((x, y))
    return outermost

def get_innermost_pixels(inverse_pixels):
    def is_valid_cc(component):
        return not component[0, 0]
    inner_circle = largest_connected_component(inverse_pixels, is_valid_cc)
    innermost = get_outermost_pixels(inner_circle)
    return innermost

def get_diameters(circle_points):
    diameters = {}
    for coord in circle_points:
        if not coord in diameters:
            max_dist = 0
            max_dist_coord = None
            for o_coord in circle_points:
                distance = dist(coord, o_coord)
                if distance > max_dist:
                    max_dist = distance
                    max_dist_coord = o_coord
            diameters[coord] = max_dist_coord
            diameters[max_dist_coord] = coord
    return diameters.items()

def get_diameter_stats(diameters):
    diameters = np.asarray([dist(x[0], x[1]) for x in diameters])
    average = np.average(diameters)
    std = np.std(diameters)
    max_ = diameters.max()
    min_ = diameters.min()
    return (average, std, max_, min_)

def get_center_point(diameters):
    centers=[]
    for pt1, pt2 in diameters:
        half_x = abs(pt1[0] - pt2[0]) / 2 + min(pt1[0], pt2[0])
        half_y = abs(pt1[1] - pt2[1]) / 2 + min(pt1[1], pt2[1])
        centers.append((half_x, half_y))
    average = [sum(x)/len(x) for x in zip(*centers)]
    return average

def set_color(array, x, y, r, g, b, w):
    half_w = w - round(w / 2)
    neg_half_w = half_w - w
    for i in range(neg_half_w, half_w):
        for j in range(neg_half_w, half_w):
            array[y + j, x + i, 0] = r
            array[y + j, x + i, 1] = g
            array[y + j, x + i, 2] = b

def color_img(original_image, data, outermost, innermost, center_point):
    image_array = np.asarray(original_image)
    for (x, y) in outermost:
        set_color(image_array, x, y, 255, 0, 0, LINE_WIDTH)
    for (x, y) in innermost:
        set_color(image_array, x, y, 0, 255, 0, LINE_WIDTH)
    center_x, center_y = round(center_point[0]), round(center_point[1])
    set_color(image_array, center_x, center_y, 0, 0, 255, POINT_WIDTH)
    return Image.fromarray(image_array)

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def process_image(filename, csv, args):
    threshold = 140 if not args.threshold else args.threshold
    invert = False if not args.invert else True
    output_dir = args.output_directory
    print("Processing " + filename + ":")
    print("Loading image...")
    img = Image.open(os.path.join(args.directory, filename))
    print("Getting " + ("light" if invert else "dark") + " pixels...")
    pixels = get_matching_pixels(img, threshold, invert)
    print("Getting " + ("dark" if invert else "light") + " pixels...")
    inverse_pixels = np.invert(pixels.copy())

    if args.debug:
        im = img_frombytes(pixels)
        im.save("dark.bmp")
        im = img_frombytes(inverse_pixels)
        im.save("light.bmp")
    
    print("Masking o-ring...")
    pixels=mask_o_ring(pixels)
    print("Finding outer edge of o-ring...")
    outermost=get_outermost_pixels(pixels)
    print("Finding inner edge of o-ring...")
    innermost=get_innermost_pixels(inverse_pixels)
    print("Getting outer diameter...")
    outermost_diameters=get_diameters(outermost)
    outer_diameter_stats=get_diameter_stats(outermost_diameters)
    print("Getting inner diameter...")
    innermost_diameters=get_diameters(innermost)
    inner_diameter_stats=get_diameter_stats(innermost_diameters)
    print("Getting center point...")
    center_point = get_center_point(outermost_diameters)
    print("Rendering witness image...")
    im = color_img(img, pixels, outermost, innermost, center_point)
    im.save(os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_processed" + ".bmp"))
    row = [ filename ] + list(outer_diameter_stats) + list(inner_diameter_stats) + [ "\n" ]
    row = [ str(v) for v in row ]
    csv.append(",".join(row))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=int, help = "RGB threshold for o-ring color")
    parser.add_argument("-d", "--directory", type=str, required=True, help = "Folder in which the o-ring images are found")
    parser.add_argument("-o", "--output-directory", type=str, required=True, help = "Folder to put processed images and csv in")
    parser.add_argument("-i", "--invert", action='store_true', help = "Expect a ligher o-ring on a darker background")
    parser.add_argument("-b", "--debug", action='store_true', help = "Outputs debugging images (dark, light) in rundir")
    args = parser.parse_args()
    csv = [ "filename, outer_diameter_avg, outer_diameter_std, outer_diameter_max, outer_diameter_min, inner_diameter_avg, inner_diameter_std, inner_diameter_max, inner_diameter_min\n" ]
    for filename in os.listdir(args.directory):
        process_image(filename, csv, args)
    f = open(os.path.join(args.output_directory, "measurements.csv"), "w")
    f.writelines(csv)
    f.close()
    
