import cv2
import numpy as np



def img_seg(img_location: str) -> list:

    image = cv2.imread(img_location)
    grey = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    ret, thresh = cv2.threshold(grey, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

    ##https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_imgproc/py_watershed/py_watershed.html



    # noise removal
    kernel = np.ones((3,3),np.uint8)
    opening = cv2.morphologyEx(thresh,cv2.MORPH_OPEN,kernel, iterations = 2)

    # sure background area
    sure_bg = cv2.dilate(opening,kernel,iterations=3)

    # Finding sure foreground area
    dist_transform = cv2.distanceTransform(opening,cv2.DIST_L2,3)
    ret, sure_fg = cv2.threshold(dist_transform,dist_transform.min(),255,0)

    # Finding unknown region
    sure_fg = np.uint8(sure_fg)
    unknown = cv2.subtract(sure_bg,sure_fg)

    # Marker labelling
    ret, markers = cv2.connectedComponents(sure_fg)

    # Add one to all labels so that sure background is not 0, but 1
    markers = markers+1

    # Now, mark the region of unknown with zero
    markers[unknown==255] = 0



    # cv2.imshow('canny edges',markers)
    # cv2.waitKey(0)

    markers = cv2.watershed(image,markers)
    image[markers == -1] = [255,0,0]

    #
    # cv2.imshow('canny edges',image)
    # cv2.waitKey(0)

    # print((markers.shape[1]))
    mark = markers.astype('uint8')
    mark = cv2.bitwise_not(mark)
    # uncomment this if you want to see how the mark
    # image looks like at that point
    cv2.imshow('Markers_v2', mark)
    cv2.waitKey(0)

    print((mark.shape))



    unique_markers = np.unique(markers)
    unique_markers = unique_markers[2:]
    # print(unique_markers)

    markers_len = len(unique_markers)

    range1 = range(markers_len)

    img = []

    for i in range1:
        marker = unique_markers[i]
        img.append(np.column_stack(np.where(markers == marker)))


    return img
