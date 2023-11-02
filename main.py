import cv2
import numpy as np

cam = cv2.VideoCapture('Lane Detection Test Video 01.mp4')

left_top_y = 0
left_top_x = 0
left_bottom_y = 0
left_bottom_x = 0
right_top_y = 0
right_top_x = 0
right_bottom_y = 0
right_bottom_x = 0
left_top_point = left_top_x, left_top_y
left_bottom_point = left_bottom_x, left_bottom_y
right_top_point = right_top_x, right_top_y
right_bottom_point = right_bottom_x, right_bottom_y


while True:
    # ex1
    ret, frame = cam.read()

    # ex2
    original_height = frame.shape[0]
    original_width = frame.shape[1]

    height = int(original_height / 4)
    width = int(original_width / 4)

    frame = cv2.resize(frame, (width, height))

    # ex3
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # ex4
    trapezoid = np.zeros((height, width), dtype=np.uint8)

    upper_left = (int(width * 0.46), int(height * 0.75))
    upper_right = (int(width * 0.54), int(height * 0.75))
    lower_left = (width * 0 , height * 1)
    lower_right = (width * 1 , height * 1)

    points_of_the_trapezoid = np.array([upper_right, upper_left, lower_left, lower_right], dtype=np.int32)

    cv2.fillConvexPoly(trapezoid, points_of_the_trapezoid, 1)
    road = trapezoid * gray

    # ex5
    points_for_stretch = np.float32(
        np.array([(width * 1, height * 0), (width * 0, height * 0), lower_left, lower_right], dtype=np.int32))
    points_of_the_trapezoid = np.float32(points_of_the_trapezoid)

    magic_matrix = cv2.getPerspectiveTransform(points_of_the_trapezoid, points_for_stretch)
    top_down = cv2.warpPerspective(road, magic_matrix, (width, height))

    # ex6
    blur = cv2.blur(top_down, ksize=(7, 7))

    # ex7
    sobel_vertical = np.float32([[-1, -2, -1],
                                 [0, 0, 0],
                                 [1, 2, 1]])

    sobel_horizontal = np.transpose(sobel_vertical)

    sobel_vertical = cv2.filter2D(np.float32(blur), -1, sobel_vertical)
    sobel_horizontal = cv2.filter2D(np.float32(blur), -1, sobel_horizontal)
    sobel = np.sqrt(sobel_vertical * sobel_vertical + sobel_horizontal * sobel_horizontal)
    sobel = cv2.convertScaleAbs(sobel)

    # ex8
    returned_by_threshold, binarized = cv2.threshold(sobel, int(255 / 5), 255, cv2.THRESH_BINARY)

    # ex9
    frame_copy = binarized.copy()

    nr_of_col = int(0.05 * frame_copy.shape[1])
    frame_copy[:, :nr_of_col] = 0
    frame_copy[:, -nr_of_col:] = 0

    indexes = np.argwhere(frame_copy > 1)
    midpoint = int(frame_copy.shape[1] / 2)
    left_indexes = indexes[indexes[:, 1] < midpoint]
    right_indexes = indexes[indexes[:, 1] >= midpoint]

    left_xs, left_ys = left_indexes[:, 1], left_indexes[:, 0]
    right_xs, right_ys = right_indexes[:, 1] - midpoint, right_indexes[:, 0]

    # ex10
    left_line = np.polynomial.polynomial.polyfit(left_xs, left_ys, deg=1)
    right_line = np.polynomial.polynomial.polyfit(right_xs, right_ys, deg=1)

    b = left_line[0]
    a = left_line[1]
    left_top_y = height
    left_top_x = (height - b) / a
    left_bottom_y = 0
    left_bottom_x = -b / a

    d = right_line[0]
    c = right_line[1]
    right_top_y = height
    right_top_x = (height - d) / c + int(width / 2)
    right_bottom_y = 0
    right_bottom_x = -d / c + int(width / 2)

    if int(width / 2) >= left_top_x >= 0 and int(width / 2) >= left_bottom_x >= 0:
        left_top_point = int(left_top_x), int(left_top_y)
        left_bottom_point = int(left_bottom_x), int(left_bottom_y)

    if width >= right_top_x >= int(width / 2) and width >= right_bottom_x >= int(width / 2):
        right_top_point = int(right_top_x), int(right_top_y)
        right_bottom_point = int(right_bottom_x), int(right_bottom_y)

    lines = cv2.line(frame_copy, left_bottom_point, left_top_point, (200, 0, 0), 10)
    lines = cv2.line(lines, right_bottom_point, right_top_point, (200, 0, 0), 10)

    #ex11
    blank = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank, left_top_point, left_bottom_point, (255, 0, 0), 3)

    magic_matrix = cv2.getPerspectiveTransform(points_for_stretch, points_of_the_trapezoid)
    final_left = cv2.warpPerspective(blank, magic_matrix, (width, height))

    left_lane = np.argwhere(final_left > 1)
    left_xs, left_ys = left_lane[:, 1], left_lane[:, 0]

    blank = np.zeros((height, width), dtype=np.uint8)
    cv2.line(blank, right_top_point, right_bottom_point, (255, 0, 0), 3)

    magic_matrix = cv2.getPerspectiveTransform(points_for_stretch, points_of_the_trapezoid)
    final_right = cv2.warpPerspective(blank, magic_matrix, (width, height))

    right_lane = np.argwhere(final_right > 1)
    right_xs, right_ys = right_lane[:, 1], right_lane[:, 0]

    final = frame.copy()
    final[final_left > 0] = (50, 50, 250)
    final[final_right > 0] = (50, 250, 50)

    if ret is False:
        break

    cv2.imshow('Original', frame)
    cv2.imshow('Grayscale', gray)
    cv2.imshow('Trapezoid', trapezoid * 255)
    cv2.imshow('Road', road)
    cv2.imshow('Top-Down', top_down)
    cv2.imshow('Blur', blur)
    cv2.imshow('Sobel', sobel)
    cv2.imshow('Binarized', binarized)
    cv2.imshow('Lines', lines)
    cv2.imshow('Final Left', final_left)
    cv2.imshow('Final Right', final_right)
    cv2.imshow('Final', final)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cam.release()
cv2.destroyAllWindows()
