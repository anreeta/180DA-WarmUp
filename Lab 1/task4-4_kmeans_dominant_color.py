'''
    References:
    * Used the following code as a baseline:
        https://code.likeagirl.io/finding-dominant-colour-on-an-image-b4e075f98097
    * Used the code snippets shown here as a guide in creating a rectangle in the center of the frame:
        https://stackoverflow.com/questions/46795669/python-opencv-how-to-draw-ractangle-center-of-image-and-crop-image-inside-rectan/46803516

    Improvements:
    * Captured video feed from webcam rather than loading in an image, similar 
        to what was done in Tasks 4.1-4.3
    * Only consider a center rectangle of the screen, rather than the entire image
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans


def find_histogram(clt):
    """
    create a histogram with k clusters
    :param: clt
    :return:hist
    """
    numLabels = np.arange(0, len(np.unique(clt.labels_)) + 1)
    (hist, _) = np.histogram(clt.labels_, bins=numLabels)

    hist = hist.astype("float")
    hist /= hist.sum()

    return hist

def plot_colors2(hist, centroids):
    bar = np.zeros((50, 300, 3), dtype="uint8")
    startX = 0

    for (percent, color) in zip(hist, centroids):
        # plot the relative percentage of each cluster
        endX = startX + (percent * 300)
        cv2.rectangle(bar, (int(startX), 0), (int(endX), 50),
                      color.astype("uint8").tolist(), -1)
        startX = endX

    # return the bar chart
    return bar

cap = cv2.VideoCapture(0) # Declare camera object


while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Convert frame to RGB
    img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Show image with rectangle around central portion of image
    ## Obtain coordinates of central rectangle (5/6 of screen)
    height, width, channels = frame.shape
    upper_left = (width // 6, height // 6)
    bottom_right = (width * 5 // 6, height * 5 // 6)
    ## Create rectangle and show image
    cv2.rectangle(frame, upper_left, bottom_right, (0, 255, 0), 2)
    cv2.imshow('Raw BGR Image',frame)

    # Perform K Means clustering
    img = img[upper_left[1]: bottom_right[1] + 1, upper_left[0]: bottom_right[0] + 1] # Crop image to central rectangle
    img = img.reshape((img.shape[0] * img.shape[1],3)) #represent as row*column,channel number
    clt = KMeans(n_clusters=3) #cluster number
    clt.fit(img)

    hist = find_histogram(clt)
    bar = plot_colors2(hist, clt.cluster_centers_)

    plt.axis("off")
    plt.imshow(bar)
    plt.show()

    # Display the resulting frame and quit if q is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()