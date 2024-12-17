import cv2
from Utility import get_rsrc_path
import os
import numpy as np

path = os.path.join(get_rsrc_path(), "image", "tiff","lena_color.tiff")
if os.path.exists(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
else:
    print("Path does not exist")
    exit(0)

# cv2.imshow("win", img)
# key = cv2.waitKey(0)
# cv2.destroyAllWindows()

# if key == ord("q"):
#     exit(0)


dst = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)


dst1 = cv2.dft(dst, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
dst1 = cv2.magnitude(dst1[:, :, 0], dst1[:, :, 1])

dst[0,0] *= 1.2

dst2 = cv2.dft(dst, flags=cv2.DFT_INVERSE | cv2.DFT_SCALE)
dst2 = cv2.magnitude(dst2[:, :, 0], dst2[:, :, 1])


show = np.concatenate([dst1, dst2], axis = 1)
print(np.max(show), np.min(show))
show = show.astype(np.uint8)

cv2.imshow("win", show)
cv2.waitKey(0)
cv2.destroyAllWindows()