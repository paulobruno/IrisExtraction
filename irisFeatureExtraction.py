import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse


# parse command line args
parser = argparse.ArgumentParser(
    description="Train an agent in a given scenario. The agent can be trained from scratch or load a trained model. Be careful: if loading a previously trained agent the scenario given should be the same, if not it could break or run as usual but with a poor performance.")

parser.add_argument(
    "image",
    help="imagem.")
parser.add_argument(
    "-p", "--plot",
    action="store_true",
    help="plot imagens.")

args = parser.parse_args()


# read the input grayscale image
src = cv2.imread(args.image, cv2.IMREAD_GRAYSCALE)


# average blur filter
avg_kernel = np.ones((5,5), np.float32) / 25
avg = cv2.filter2D(src, -1, avg_kernel)

if args.plot:
    cv2.imshow('average blur', avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# binary threshold
_, bin_avg = cv2.threshold(avg, 50, 255, cv2.THRESH_BINARY_INV)

if args.plot:
    cv2.imshow('binary image', bin_avg)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_avg, 8, cv2.CV_32S)

# TODO: compute a colormap for the label's image and replace matplotlib here
if args.plot:
    plt.imshow(labels)
    plt.show()


# extract the pupil

# get the size from all connected components
# in the stats, the size is the last column, so we take the -1 column
cc_sizes = np.array(stats)[:, -1]

# the connected components function also counts the background
# thus, in this case, the background will have the max size
# the pupil will have the second highest size
# to get the index of the pupil, we take the second greater connected component
pupil_idx = np.argsort(cc_sizes)[-2]

# get pupil coords and radius
pupil_w = stats[pupil_idx, cv2.CC_STAT_WIDTH]
pupil_radius = pupil_w / 2

pupil_cx = int(centroids[pupil_idx, 0])
pupil_cy = int(centroids[pupil_idx, 1])

# create pupil mask, a black and white image with only the pupil region
pupil_mask = np.zeros(shape=(src.shape + (3,)), dtype=np.uint8)
cv2.circle(pupil_mask, (pupil_cx, pupil_cy), int(pupil_radius), (255, 255, 255), cv2.FILLED)

# draw a circle aroung the pupil
pupil_draw = cv2.cvtColor(src, cv2.COLOR_GRAY2RGB)
cv2.circle(pupil_draw, (pupil_cx, pupil_cy), int(pupil_radius), (0, 0, 255), 2)

if args.plot:
    cv2.imshow('pupil circle', pupil_draw)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

cv2.imwrite('out_pupil.jpg', pupil_draw)


# extract iris outer edge

# smooth the image to better identify the iris
smoothed = src
for i in range(200):
    smoothed = cv2.medianBlur(smoothed, 5)

if args.plot:
    cv2.imshow('smoothed', smoothed)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# detect all circles using Hough's circle transform
circles = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, 1, 20, 
    param1=50, param2=30, minRadius=int(pupil_radius+10), maxRadius=int(pupil_radius*10))

# convert float values to int
circles = np.uint16(np.around(circles))

if args.plot:
    hough_circles = cv2.cvtColor(smoothed, cv2.COLOR_GRAY2BGR)
    for c in circles[0]:
        cv2.circle(hough_circles, (c[0], c[1]), c[2], (0, 0, 255), 2)

    cv2.imshow('hough circles', hough_circles)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# the transform will detect multiple circles, however OpenCV's HoughCircles
#   sorts the results by 'confidence', therefore we want the first one
iris_cx, iris_cy, iris_radius = circles[0,0,0], circles[0,0,1], circles[0,0,2]

# draw a binary image of only the iris region
iris_region = np.zeros(shape=(src.shape + (3,)), dtype=np.uint8)
cv2.circle(iris_region, (iris_cx, iris_cy), iris_radius, (255, 255, 255), cv2.FILLED)

if args.plot:
    iris_plot = pupil_draw.copy()
    cv2.circle(iris_plot, (iris_cx, iris_cy), iris_radius, (255, 0, 0), 2)

    cv2.imshow('iris and pupil detected', iris_plot)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# show only the iris region (excluding the pupil also)
colored_img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
iris = cv2.bitwise_and(iris_region, colored_img)
iris = cv2.bitwise_and(iris, cv2.bitwise_not(pupil_mask))

if args.plot:
    cv2.imshow('iris', iris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# bounding box of the iris
# center x is index 0, center y is index 1, and radius is index 2 
x_0 = iris_cx - iris_radius
x_t = iris_cx + iris_radius
y_0 = iris_cy - iris_radius
y_t = iris_cy + iris_radius

iris_bb = iris[y_0:y_t, x_0:x_t]

if args.plot:
    cv2.imshow('iris bounding box', iris_bb)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# normalization
def daugman_normalization(image, height, width, r_in, r_out):       # Daugman归一化，输入为640*480,输出为width*height
    thetas = np.arange(0, 2 * np.pi, 2 * np.pi / width)  # Theta values
    print(thetas)
    print(r_out)
    #r_out = r_in + r_out
    print(r_in)
    print(r_out)
    # Create empty flatten image
    flat = np.zeros((height, width, 3), np.uint8)
    circle_x = int(image.shape[0] // 2)
    circle_y = int(image.shape[1] // 2)
    print(circle_x)
    print(circle_y)
    print()

    for i in range(width):
        for j in range(height):
            theta = thetas[i]  # value of theta coordinate
            r_pro = j / height  # value of r coordinate(normalized)
            
            # get coordinate of boundaries
            Xi = circle_x + r_in * np.cos(theta)
            Yi = circle_y + r_in * np.sin(theta)
            Xo = circle_x + r_out * np.cos(theta)
            Yo = circle_y + r_out * np.sin(theta)

            # the matched cartesian coordinates for the polar coordinates
            Xc = (1 - r_pro) * Xi + r_pro * Xo
            #print(r_pro)
            #print(Xi)
            #print(Xo)
            #print(Yi)
            #print(Yo)
            #print()
            Yc = (1 - r_pro) * Yi + r_pro * Yo

            color = image[int(Xc)][int(Yc)]  # color of the pixel

            flat[j][i] = color
    return flat  # liang
    

print(iris_bb.shape)

normal = daugman_normalization(iris_bb, 60, 360, pupil_radius, iris_radius)

if args.plot:
    cv2.imshow('normalized', normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
normal = cv2.equalizeHist(normal)

if args.plot:
    cv2.imshow('normalized', normal)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
