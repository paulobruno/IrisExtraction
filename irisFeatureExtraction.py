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
avg_kernel = np.ones((5,5), np.float32)/25
avg = cv2.filter2D(src, -1, avg_kernel)

if args.plot:
    plt.subplot(132)
    plt.imshow(avg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()
    

# binary threshold
_, bin_avg = cv2.threshold(avg, 50, 255, cv2.THRESH_BINARY_INV)

if args.plot:
    plt.subplot(132)
    plt.imshow(bin_avg, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    plt.show()


# connected components
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(bin_avg, 8, cv2.CV_32S)

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

# get pupil bounding box
pupil_x = stats[pupil_idx, cv2.CC_STAT_LEFT]
pupil_y = stats[pupil_idx, cv2.CC_STAT_TOP]
pupil_w = stats[pupil_idx, cv2.CC_STAT_WIDTH]
pupil_h = stats[pupil_idx, cv2.CC_STAT_HEIGHT]

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
    plt.imshow(pupil_draw)
    plt.show()

cv2.imwrite('out_pupil.jpg', pupil_draw)


# extract iris outer edge

canny_img = cv2.Canny(src, 50, 50)

smoothed = src
for i in range(200):
    smoothed = cv2.medianBlur(smoothed, 5)

edges = cv2.Canny(smoothed, 50, 50)

if args.plot:
    plt.imshow(canny_img, cmap='gray')
    plt.show()
    
    plt.subplot(121)
    plt.imshow(smoothed, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.subplot(122)
    plt.imshow(edges, cmap='gray')
    plt.xticks([])
    plt.yticks([])
    
    plt.show()
    

circles = cv2.HoughCircles(smoothed, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=int(pupil_radius+10), maxRadius=int(pupil_radius*10))

circles = np.uint16(np.around(circles))
i = circles[0,0]

iris_region = np.zeros(shape=(src.shape + (3,)), dtype=np.uint8)
cv2.circle(iris_region, (i[0],i[1]), i[2], (255,255,255), cv2.FILLED)

iris_src = pupil_draw.copy()
cv2.circle(iris_src, (i[0],i[1]), i[2], (255,0,0), 2)

if args.plot:
    cv2.imshow('detected circles', iris_src)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


colored_img = cv2.cvtColor(src, cv2.COLOR_GRAY2BGR)
iris = cv2.bitwise_and(iris_region, colored_img)
iris = cv2.bitwise_and(iris, cv2.bitwise_not(pupil_mask))

if args.plot:
    cv2.imshow('iris', iris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


x_0 = i[0] - i[2]
x_t = i[0] + i[2]
y_0 = i[1] - i[2]
y_t = i[1] + i[2]

so_iris = iris[y_0:y_t, x_0:x_t]

if args.plot:
    cv2.imshow('so_iris', so_iris)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


img_cut = src[y_0:y_t, x_0:x_t]

if args.plot:
    cv2.imshow('cut', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

edges = cv2.Canny(img_cut,50,150,apertureSize = 3)
if args.plot:
    cv2.imshow('edges', edges)
    cv2.waitKey(0)

lines = cv2.HoughLinesP(edges, 1, np.pi/180, 100, minLineLength=50, maxLineGap=10)
if (lines):
    for line in lines:
        x1,y1,x2,y2 = line[0]
        cv2.line(img_cut,(x1,y1),(x2,y2),(0,255,0),2)

if args.plot:
    cv2.imshow('lines', img_cut)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


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
    

print(so_iris.shape)

normal = daugman_normalization(so_iris, 60, 360, pupil_radius, i[2])

if args.plot:
    plt.imshow(normal)
    plt.show()

normal = cv2.cvtColor(normal, cv2.COLOR_BGR2GRAY)
normal = cv2.equalizeHist(normal)

if args.plot:
    plt.imshow(normal, cmap='gray')
    plt.show()
