from SIFT import SIFT
import cv2

sift = SIFT()

img = cv2.imread('../dataset/img1.ppm')
sift.create_initial_image(img)
sift.build_gaussian_pyramid()
sift.build_dog_pyramid()
sift.find_scale_space_extrema()
sift.calc_descriptors()

# for octave, layers in sift.gaussian_pyramid.items():
#     for i, layer in enumerate(layers):
#         print(len(layers))
#         cv2.imshow(str(octave)+":"+str(i), layer)

# for octave, layers in sift.dog_pyramid.items():
#     for i, layer in enumerate(layers):
#         print(len(layers))
#         cv2.imshow(str(octave)+":"+str(i), layer)

cv2.waitKey(0)
