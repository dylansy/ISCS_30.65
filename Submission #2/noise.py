import cv2
import numpy as np

def add_poisson(img, lambda_val):
    poisson_noise = np.random.poisson(lambda_val, size=img.shape)

    noisy_image = img + poisson_noise

    if noisy_image.dtype != np.uint8:
        noisy_image = cv2.convertScaleAbs(noisy_image)

    return noisy_image

def add_salt(img, ratio):
    noisy_mask = np.random.rand(*img.shape) < ratio

    img[noisy_mask] = 255

    return img

img = cv2.imread("test.png", 0)

noisy_salt_image = add_salt(img.copy(), 0.05)
noisy_poisson_image = add_poisson(img.copy(), 25)

cv2.imshow("original", img)
cv2.imshow("salty image", noisy_salt_image)
cv2.imshow("poisson image", noisy_poisson_image)

cv2.waitKey(0)

cv2.destroyAllWindows()
