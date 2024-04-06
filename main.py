import cv2
import numpy as np

my_photo = cv2.imread('C:/Users/asonza/Downloads/dataroad1.png')

# Преобразование изображения в черно-белое
bw_image = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)

# Применение медианного фильтра с параметром 10
median_image = cv2.medianBlur(bw_image, 5)
# Применение бинаризации для выделения контуров
_, binary_image = cv2.threshold(median_image, 0, 255, cv2.THRESH_BINARY)

# Нахождение контуров
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Создание пустого изображения для контуров
contour_image = np.zeros_like(my_photo)

# Нарисовать контуры на пустом изображении
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 2)

# Отображение исходного, медианно отфильтрованного и контурного изображений
cv2.imshow('Original Image', my_photo)
cv2.imshow('Median Filtered Image', median_image)
cv2.imshow('Contours', contour_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()