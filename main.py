import cv2
import numpy as np

# Загрузка изображения
my_photo = cv2.imread('C:/Users/asonza/Downloads/dataroad1.png')

# Преобразование изображения в черно-белое
bw_image = cv2.cvtColor(my_photo, cv2.COLOR_BGR2GRAY)

# Применение уравнивания гистограммы для увеличения контрастности
kernel = np.array([[-1,0,1], [-2,0,2], [-1,0,1]])
equalized_image = cv2.filter2D(bw_image, -1, kernel)

# Применение медианного фильтра с параметром 5
median_image = cv2.medianBlur(equalized_image, 15)

# Применение бинаризации для выделения контуров
_, binary_image = cv2.threshold(median_image, 20, 255, cv2.THRESH_BINARY)

# Нахождение контуров
contours, _ = cv2.findContours(binary_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

epsilon = 0.01 * cv2.arcLength(contours[0], True)
approx = cv2.approxPolyDP(contours[3], epsilon, True)

# Создание пустого изображения для контуров
contour_image = np.zeros_like(my_photo)

# Нарисовать контуры на пустом изображении
cv2.drawContours(contour_image, contours, -1, (255, 255, 255), 1)
#cv2.drawContours(contour_image, [approx], -1, (255, 255, 255), 1)

# Отображение исходного, уравненного, медианно отфильтрованного и контурного изображений
cv2.imshow('Original Image', my_photo)
cv2.imshow('Equalized Image', equalized_image)
# cv2.imshow('Median Filtered Image', median_image)
cv2.imshow('Contours', contour_image)

# Ожидание нажатия клавиши для закрытия окон
cv2.waitKey(0)
cv2.destroyAllWindows()
