
import numpy as np
import cv2
import matplotlib.pyplot as plt

# Black = 1, White = -1
filter1 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])  # 4x10

filter2 = np.array([[1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
                    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]])  # 10x18

filter3 = np.array([[1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1],
                    [1, 1, 1, 1, 1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 1]])  # 15x15

# Face Image Code --------------------------------------------------------------------------------------------------------------------------------------------

faceImg = cv2.imread('D:\\FCI\\lvl 4\\2nd term\\Computer Vision\\assignments\\assignment 3\\face.jpg')
faceImgGray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

resizedImg = cv2.resize(faceImgGray, (24, 24))

imgHeight, imgWidth = resizedImg.shape

integralImg = np.zeros((imgHeight, imgWidth))

resultImg1 = np.zeros((imgHeight, imgWidth))
resultImg2 = np.zeros((imgHeight, imgWidth))
resultImg3 = np.zeros((imgHeight, imgWidth))

Threshold = 253

# calculate the integral image
for i in range(imgHeight):
    for j in range(imgWidth):
        window = faceImgGray[0:i + 1, 0:j + 1]  # i:i+4, j:j+10
        integralImg[i, j] = np.sum(window)

f = open("Threshold.txt", "a")  # open a file to write in ... "a" = append mode
f.write("Threshold = " + str(Threshold))
f.close()

# Filter 1
for h in range(21):
    for w in range(15):

        # Black Part
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter1.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+1, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + int(filterHeight / 2) - 1, w + filterWidth - 1]  # integralImg[h+1, w+9]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = corner  # Value of the black corner --> integralImg[h+1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+3, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+3, w+9]

        sumWhite = corner - above - beside + diagonal

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("Face - Feature-1 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg1[h, w] = int(originPixelValue)

# Filter 2
for h in range(15):
    for w in range(7):

        # Black Part
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter2.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+17]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+4, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + int(filterHeight / 2) - 1, w + filterWidth - 1]  # integralImg[h+4, w+17]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = corner  # Value of the black corner --> integralImg[h+4, w+17]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+9, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+4, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+9, w+17]

        sumWhite = corner - above - beside + diagonal

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("Face - Feature-2 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg2[h, w] = int(originPixelValue)

# Filter 3
for h in range(10):
    for w in range(10):

        # Black Part 1
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter3.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + int(filterWidth / 3) - 1]  # integralImg[h-1, w+4]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+14, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + int(filterWidth / 3) - 1]  # integralImg[h+14, w+4]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + int(filterWidth / 3) + 4]  # integralImg[h-1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = corner  # black 1 corner --> integralImg[h+14, w+4]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w + int(filterWidth / 3) - 1]  # integralImg[h-1, w+4]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + int(filterWidth / 3) + 4]  # integralImg[h+14, w+9]

        sumWhite = corner - above - beside + diagonal

        # Black Part 2
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter3.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+14]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = corner  # white corner --> integralImg[h+14, w+9]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w + int(filterWidth / 3) + 4]  # integralImg[h-1, w+9]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+14, w+14]

        sumBlack = sumBlack + (corner - above - beside + diagonal)  # black part 1&2 sum

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("Face - Feature-3 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg3[h, w] = int(originPixelValue)

fig, image = plt.subplots(2, 3)
fig.suptitle('Assignment 3')
image[0, 0].set_title("Original Image")
image[0, 0].imshow(faceImgGray)
image[0, 1].set_title("Resized Image")
image[0, 1].imshow(resizedImg)
image[0, 2].set_title("Integral Image")
image[0, 2].imshow(integralImg)
image[1, 0].set_title("Feature 1 - Output")
image[1, 0].imshow(resultImg1)
image[1, 1].set_title("Feature 2 - Output")
image[1, 1].imshow(resultImg2)
image[1, 2].set_title("Feature 3 - Output")
image[1, 2].imshow(resultImg3)

plt.show()

# END Face Image Code ----------------------------------------------------------------------------------------------------------------------------------------

# NON-Face Code ----------------------------------------------------------------------------------------------------------------------------------------------

faceImg = cv2.imread('D:\\FCI\\lvl 4\\2nd term\\Computer Vision\\assignments\\assignment 3\\non-face.jpg')
faceImgGray = cv2.cvtColor(faceImg, cv2.COLOR_BGR2GRAY)

resizedImg = cv2.resize(faceImgGray, (24, 24))

imgHeight, imgWidth = resizedImg.shape

integralImg = np.zeros((imgHeight, imgWidth))

resultImg1 = np.zeros((imgHeight, imgWidth))
resultImg2 = np.zeros((imgHeight, imgWidth))
resultImg3 = np.zeros((imgHeight, imgWidth))

# calculate the integral image
for i in range(imgHeight):
    for j in range(imgWidth):
        window = faceImgGray[0:i + 1, 0:j + 1]  # i:i+4, j:j+10
        integralImg[i, j] = np.sum(window)

# Filter 1
for h in range(21):
    for w in range(15):

        # Black Part
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter1.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+1, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + int(filterHeight / 2) - 1, w + filterWidth - 1]  # integralImg[h+1, w+9]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = corner  # Value of the black corner --> integralImg[h+1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+3, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+3, w+9]

        sumWhite = corner - above - beside + diagonal

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("NON-Face - Feature-1 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg1[h, w] = int(originPixelValue)

# Filter 2
for h in range(15):
    for w in range(7):

        # Black Part
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter2.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+17]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+4, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + int(filterHeight / 2) - 1, w + filterWidth - 1]  # integralImg[h+4, w+17]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = corner  # Value of the black corner --> integralImg[h+4, w+17]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+9, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h + int(filterHeight / 2) - 1, w - 1]  # integralImg[h+4, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+9, w+17]

        sumWhite = corner - above - beside + diagonal

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("NON-Face - Feature-2 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg2[h, w] = int(originPixelValue)

# Filter 3
for h in range(10):
    for w in range(10):

        # Black Part 1
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter3.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + int(filterWidth / 3) - 1]  # integralImg[h-1, w+4]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = integralImg[h + filterHeight - 1, w - 1]  # integralImg[h+14, w-1]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w - 1]  # integralImg[h-1, w-1]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + int(filterWidth / 3) - 1]  # integralImg[h+14, w+4]

        sumBlack = corner - above - beside + diagonal  # black part sum

        # White Part
        above = 0
        beside = 0
        diagonal = 0

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + int(filterWidth / 3) + 4]  # integralImg[h-1, w+9]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = corner  # black 1 corner --> integralImg[h+14, w+4]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w + int(filterWidth / 3) - 1]  # integralImg[h-1, w+4]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + int(filterWidth / 3) + 4]  # integralImg[h+14, w+9]

        sumWhite = corner - above - beside + diagonal

        # Black Part 2
        above = 0
        beside = 0
        diagonal = 0
        filterHeight, filterWidth = filter3.shape

        # Check the current pixel coordinates in the integral image
        if h != 0 or w != 0:

            # Check if the height is = 0 (for above pixel calculation)
            if h == 0:
                above = 0
            else:
                above = integralImg[h - 1, w + filterWidth - 1]  # integralImg[h-1, w+14]

            # Check if the width is = 0 (for beside pixel calculation)
            if w == 0:
                beside = 0
            else:
                beside = corner  # white corner --> integralImg[h+14, w+9]

            # Check if both are != 0 (for diagonal pixel calculation)
            if h != 0 and w != 0:
                diagonal = integralImg[h - 1, w + int(filterWidth / 3) + 4]  # integralImg[h-1, w+9]
            else:
                diagonal = 0

        corner = integralImg[h + filterHeight - 1, w + filterWidth - 1]  # integralImg[h+14, w+14]

        sumBlack = sumBlack + (corner - above - beside + diagonal)  # black part 1&2 sum

        originPixelValue = sumBlack - sumWhite

        if originPixelValue >= Threshold:
            f = open("NON-Face - Feature-3 results.txt", "a")
            f.write("(" + str(h) + ", " + str(w) + ")\n")
            f.close()

        resultImg3[h, w] = int(originPixelValue)

fig, image = plt.subplots(2, 3)
fig.suptitle('Assignment 3')
image[0, 0].set_title("Original Image")
image[0, 0].imshow(faceImgGray)
image[0, 1].set_title("Resized Image")
image[0, 1].imshow(resizedImg)
image[0, 2].set_title("Integral Image")
image[0, 2].imshow(integralImg)
image[1, 0].set_title("Feature 1 - Output")
image[1, 0].imshow(resultImg1)
image[1, 1].set_title("Feature 2 - Output")
image[1, 1].imshow(resultImg2)
image[1, 2].set_title("Feature 3 - Output")
image[1, 2].imshow(resultImg3)

plt.show()

# END NON-Face Code ------------------------------------------------------------------------------------------------------------------------------------------