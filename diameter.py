from PIL import Image
import sys
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import threshold_niblack
from math import dist
from collections import deque
import argparse
import os

LINE_WIDTH=6
POINT_WIDTH=20

def is_matching_pixel(rgb, threshold, invert, yellow):
    r, g, b = rgb
    if invert:
        if yellow:
            return b > threshold and g < 150 and r < 150
        else:
            return r > threshold and g > threshold and b > threshold
    else:
        if yellow:
            return b < threshold and g > 150 and r > 150
        else:
            return r < threshold and g < threshold and b < threshold

def get_matching_pixels(img, pixels, threshold, invert, yellow):
    matching_pixels = np.full((img.size[1], img.size[0]), False, dtype=bool)
    for x in range(img.size[0]):
        for y in range(img.size[1]):
            if is_matching_pixel (pixels[x, y], threshold, invert, yellow):
                matching_pixels[y, x] = True
            else:
                matching_pixels[y, x] = False
    return matching_pixels

def traverse_component(pixels, visited_set, coords, is_node):
    component = set()
    to_visit = deque([coords])
    def visit ():
        visited_set.add((x, y))
        component.add((x, y))
    while True:
        if not to_visit:
            return component
        x, y = to_visit.pop()
        visit ()
        if not (x, y - 1) in visited_set and y - 1 > 0 and is_node(pixels, y - 1, x):
            to_visit.append((x, y-1))
        if not (x, y + 1) in visited_set and y + 1 < len(pixels) and is_node(pixels, y + 1, x):
            to_visit.append((x, y+1))
        if not (x - 1, y) in visited_set and x - 1 > 0 and is_node(pixels, y, x - 1):
            to_visit.append((x-1, y))
        if not (x + 1, y) in visited_set and x + 1 < len(pixels[0]) and is_node(pixels, y, x + 1):
            to_visit.append((x+1, y))
    return component

def largest_connected_component(pixels, is_node, is_valid_cc):
    visited_set = set()
    largest_cc = None
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
            if is_node(pixels, y, x) and not (x, y) in visited_set:
                component = traverse_component(pixels, visited_set, (x, y), is_node)
                if is_valid_cc(component) and (largest_cc == None or len(component) > len(largest_cc)):
                    largest_cc = component
    return largest_cc

def mask_o_ring(pixels):
    visited_set = set()
    def is_node(pixels, y, x):
        return pixels[y][x] 
    largest_cc = largest_connected_component(pixels, is_node, lambda x: True)
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
            if not ((x, y) in largest_cc):
                pixels[y, x] = False

    

def get_outermost_pixels(pixels):
    max_y = [None] * len(pixels[0])
    max_x = [None] * len(pixels)
    min_y = [None] * len(pixels[0])
    min_x = [None] * len(pixels)
    for y in range(len(pixels)):
        for x in range(len(pixels[0])):
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

def get_innermost_pixels(pixels):
    visited_set = set()
    def is_node(pixels, y, x):
        return not pixels[y][x]
    def is_valid_cc(component):
        return not (0, 0) in component
    largest_cc = largest_connected_component(pixels, is_node, is_valid_cc)
    image_array = np.full((len(pixels), len(pixels[0])), False, dtype=bool)
    for x, y in largest_cc:
        image_array[y, x] = True
    innermost = get_outermost_pixels(image_array)
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

def get_average_diameter(diameters):
    average = sum([dist(x[0], x[1]) for x in diameters])/len(diameters)
    return average
    
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

def get_ruler_line(pixels, threshold, ruler_line_height):
    def is_node(pixels, y, x):
        return pixels[y][x][0] < threshold and pixels[y][x][1] < threshold and pixels[y][x][2] < threshold
    def is_valid_cc(component):
        min_y = min([y[1] for y in component])
        max_y = max([y[1] for y in component])
        return max_y - min_y < ruler_line_height
    cc = largest_connected_component(pixels, is_node, is_valid_cc)
    return cc

def get_measurement_points(pixels, threshold, ruler_line):
    def is_valid(y, x):
        return pixels[y][x][0] < threshold and pixels[y][x][1] < threshold and pixels[y][x][2] < threshold
    min_y = min([y[1] for y in ruler_line])
    max_y = max([y[1] for y in ruler_line])
    min_x = min([x[0] for x in ruler_line])
    line_point = (min_x + 10, min_y)
    for i in range (1, 100):
        x, y = line_point
        if is_valid(y - i, x):
            return (line_point, (x, y-(i+(round((3/5)*i)))))
    line_point = (min_x + 10, max_y)
    for i in range (1, 100):
        x, y = line_point
        if is_valid(y + i, x):
            return (line_point, (x, y+(i+(round((3/5)*i)))))
    return None

def get_px_per_in(measurement_points):
    pixels = abs(measurement_points[0][1] - measurement_points[1][1])
    return pixels * 64

def ordered_measurement_points(points):
    if points[0][1] <= points[1][1]:
        return points
    return (points[1], points[0])

def color_img(original_image, data, outermost, innermost, center_point, ruler_line, measurement_points):
    image_array = np.asarray(original_image)
    size = data.shape[::-1]
    for (x, y) in outermost:
        set_color(image_array, x, y, 255, 0, 0, LINE_WIDTH)
    for (x, y) in innermost:
        set_color(image_array, x, y, 0, 255, 0, LINE_WIDTH)
    for (x, y) in ruler_line:
        set_color(image_array, x, y, 255, 0, 0, 1)
    measure1, measure2 = ordered_measurement_points(measurement_points)
    for i in range(measure2[1] - measure1[1]):
        set_color(image_array, measure1[0], measure1[1] + i, 0, 255, 0, LINE_WIDTH)
    center_x, center_y = round(center_point[0]), round(center_point[1])
    set_color(image_array, center_x, center_y, 0, 0, 255, POINT_WIDTH)
    return Image.fromarray(image_array)

def img_frombytes(data):
    size = data.shape[::-1]
    databytes = np.packbits(data, axis=1)
    return Image.frombytes(mode='1', size=size, data=databytes)

def process_image(filename, csv, args):
    threshold = 140 if not args.threshold else args.threshold
    ruler_threshold = 100 if not args.ruler_threshold else args.ruler_threshold
    invert = False if not args.invert else True
    yellow = False if not args.yellow else True
    ruler_line_height = 100 if not args.ruler_line_height else args.ruler_line_height
    output_dir = args.output_directory
    print("Processing " + filename + ":")
    print("Loading image...")
    img = Image.open(args.directory + "/" + filename)
    pixels = img.load()
    print("Getting " + ("light" if invert else "dark") + " pixels...")
    pixels = get_matching_pixels(img, pixels, threshold, invert, yellow)

    im = img_frombytes(pixels)
    im.save("light.bmp")
    
    print("Finding a line on the ruler...")
    raw_image=np.asarray(img)
    ruler_line = get_ruler_line(raw_image, ruler_threshold, ruler_line_height)
    print("Getting px/in conversion...")
    measurement_points = get_measurement_points(raw_image, ruler_threshold, ruler_line)
    px_per_in = get_px_per_in(measurement_points)
    print("Px/in: " + str(px_per_in))
    print("Masking o-ring...")
    mask_o_ring(pixels)
    print("Finding outer edge of o-ring...")
    outermost=get_outermost_pixels(pixels)
    print("Finding inner edge of o-ring...")
    innermost=get_innermost_pixels(pixels)
    print("Getting outer diameter...")
    outermost_diameters=get_diameters(outermost)
    outer_diameter=get_average_diameter(outermost_diameters)
    print("Getting inner diameter...")
    innermost_diameters=get_diameters(innermost)
    inner_diameter=get_average_diameter(innermost_diameters)
    print("Getting center point...")
    center_point = get_center_point(outermost_diameters)
    print("Rendering witness image...")
    im = color_img(img, pixels, outermost, innermost, center_point, ruler_line, measurement_points)
    im.save(output_dir + "/" + filename.rsplit('.', 1)[0] + "_processed" + ".bmp")
    csv.append(filename + "," + str(outer_diameter) + "," + str(outer_diameter / px_per_in) + "," + str(inner_diameter) + "," + str(inner_diameter / px_per_in) + "," + str(px_per_in) + "\n")

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=int, help = "RGB threshold for o-ring color")
    parser.add_argument("-r", "--ruler-threshold", type=int, help = "RGB threshold for rule color")
    parser.add_argument("-d", "--directory", type=str, required=True, help = "Folder in which the o-ring images are found")
    parser.add_argument("-o", "--output-directory", type=str, required=True, help = "Folder to put processed images and csv in")
    parser.add_argument("-i", "--invert", action='store_true', help = "Expect a ligher o-ring on a darker background")
    parser.add_argument("-y", "--yellow", action='store_true', help = "Expect a yellow o-ring")
    parser.add_argument("-l", "--ruler-line-height", type=int, help = "Limit on the height of a ruler line")
    args = parser.parse_args()
    csv = [ "filename, outer_diameter_px, outer_diameter_in, inner_diameter_px, inner_diameter_in, px_per_in\n" ]
    for filename in os.listdir(args.directory):
        process_image(filename, csv, args)
    f = open(args.output_directory + "/" + "measurements.csv", "w")
    f.writelines(csv)
    f.close()
    
