import cv2


def get_bw_image(path, name):
    image = cv2.imread(f'{path}/{name}')
    bw_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f'{path}/bw_image.png', bw_image)
