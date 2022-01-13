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

def get_diameter_stats_in(diameters, px_per_in):
    diameters = np.asarray([dist(x[0], x[1]) / px_per_in for x in diameters])
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

def get_ruler_line(img, ruler_pixels, ruler_line_height):
    def is_valid_cc(component):
        component = np.transpose(component.nonzero())
        min_y = min([y[0] for y in component])
        max_y = max([y[0] for y in component])
        return max_y - min_y < ruler_line_height
    cc = largest_connected_component(ruler_pixels, is_valid_cc)
    cc = np.transpose(cc.nonzero())
    ruler_line = [ (x[1], x[0]) for x in cc]
    return set(ruler_line)

def get_points_between_lines(pixels, ruler_line):
    min_y = min([y[1] for y in ruler_line])
    max_y = max([y[1] for y in ruler_line])
    min_x = min([x[0] for x in ruler_line])
    line_point = (min_x + 20, min_y)
    for i in range (1, 100):
        x, y = line_point
        if pixels[y - i, x]:
            return (line_point, (x, y - i))
    line_point = (min_x + 20, max_y)
    for i in range (1, 100):
        x, y = line_point
        if pixels[y + i, x]:
            return (line_point, (x, y - i))
    return None

def get_measurement_points(pixels, ruler_line):
    p1, p2 = get_points_between_lines(pixels, ruler_line)
    pixels_between = abs(p1[1] - p2[1])
    min_line_height = round(pixels_between * (2/5))
    max_line_height = pixels_between
    if p1[1] > p2[1]:
        x, y = p2
        for i in range (min_line_height, max_line_height):
            if not pixels[y - i, x]:
                return (p1, (x, y - i))
        return (p1, (x, y - max_line_height))
    else:
        x, y = p2
        for i in range (min_line_height, max_line_height):
            if not pixels[y + i, x]:
                return (p1, (x, y + i))
        return (p1, (x, y + max_line_height))   

def get_px_per_in(measurement_points):
    pixels = abs(measurement_points[0][1] - measurement_points[1][1])
    return pixels * 64

def ordered_measurement_points(points):
    if points[0][1] <= points[1][1]:
        return points
    return (points[1], points[0])

def set_color(array, x, y, r, g, b, w):
    half_w = w - round(w / 2)
    neg_half_w = half_w - w
    for i in range(neg_half_w, half_w):
        for j in range(neg_half_w, half_w):
            try:
                array[y + j, x + i, 0] = r
                array[y + j, x + i, 1] = g
                array[y + j, x + i, 2] = b
            except:
                pass

def color_img(original_image, data, outermost, innermost, center_point, ruler_line, measurement_points):
    image_array = np.asarray(original_image)
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
    ruler_line_height = 100 if not args.ruler_line_height else args.ruler_line_height
    output_dir = args.output_directory
    print("Processing " + filename + ":")
    print("Loading image...")
    img = Image.open(os.path.join(args.directory, filename))
    print("Getting " + ("light" if invert else "dark") + " pixels...")
    pixels = get_matching_pixels(img, threshold, invert)
    print("Getting " + ("dark" if invert else "light") + " pixels...")
    inverse_pixels = np.invert(pixels.copy())
    print("Getting ruler line pixels...")
    ruler_pixels = get_matching_pixels(img, ruler_threshold, False)

    if args.debug:
        im = img_frombytes(pixels)
        im.save("dark.bmp")
        im = img_frombytes(inverse_pixels)
        im.save("light.bmp")

    print("Finding a line on the ruler...")
    ruler_line = get_ruler_line(img, ruler_pixels, ruler_line_height)
    print("Getting px/in conversion...")
    measurement_points = get_measurement_points(ruler_pixels, ruler_line)
    px_per_in = get_px_per_in(measurement_points)
    print("Px/in: " + str(px_per_in))
    print("Masking o-ring...")
    pixels=mask_o_ring(pixels)
    print("Finding outer edge of o-ring...")
    outermost=get_outermost_pixels(pixels)
    print("Finding inner edge of o-ring...")
    innermost=get_innermost_pixels(inverse_pixels)
    print("Getting outer diameter...")
    outermost_diameters=get_diameters(outermost)
    outer_diameter_stats=get_diameter_stats(outermost_diameters)
    outer_diameter_stats_in=get_diameter_stats_in(outermost_diameters, px_per_in)
    print("Getting inner diameter...")
    innermost_diameters=get_diameters(innermost)
    inner_diameter_stats=get_diameter_stats(innermost_diameters)
    inner_diameter_stats_in=get_diameter_stats_in(innermost_diameters, px_per_in)
    print("Getting center point...")
    center_point = get_center_point(outermost_diameters)
    print("Rendering witness image...")
    im = color_img(img, pixels, outermost, innermost, center_point, ruler_line, measurement_points)
    im.save(os.path.join(output_dir, filename.rsplit('.', 1)[0] + "_processed" + ".bmp"))
    row = [ filename ] + list(outer_diameter_stats) + list(outer_diameter_stats_in) + list(inner_diameter_stats) + list(inner_diameter_stats_in) + [ "\n" ]
    row = [ str(v) for v in row ]
    csv.append(",".join(row))

if __name__== "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--threshold", type=int, help = "RGB threshold for o-ring color")
    parser.add_argument("-r", "--ruler-threshold", type=int, help = "RGB threshold for rule color")
    parser.add_argument("-d", "--directory", type=str, required=True, help = "Folder in which the o-ring images are found")
    parser.add_argument("-o", "--output-directory", type=str, required=True, help = "Folder to put processed images and csv in")
    parser.add_argument("-i", "--invert", action='store_true', help = "Expect a ligher o-ring on a darker background")
    parser.add_argument("-l", "--ruler-line-height", type=int, help = "Limit on the height of a ruler line")
    parser.add_argument("-b", "--debug", action='store_true', help = "Outputs debugging images (dark, light) in rundir")
    args = parser.parse_args()
    csv = [ "filename, outer_diameter_px_avg, outer_diameter_px_std, outer_diameter_px_max, outer_diameter_px_min, outer_diameter_in_avg, outer_diameter_in_std, outer_diameter_in_max, outer_diameter_in_min, inner_diameter_px_avg, inner_diameter_px_std, inner_diameter_px_max, inner_diameter_px_min, inner_diameter_in_avg, inner_diameter_in_std, inner_diameter_in_max, inner_diameter_in_min\n" ]
    for filename in os.listdir(args.directory):
        process_image(filename, csv, args)
    f = open(os.path.join(args.output_directory, "measurements.csv"), "w")
    f.writelines(csv)
    f.close()
    
