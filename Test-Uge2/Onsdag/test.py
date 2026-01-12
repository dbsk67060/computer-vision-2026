import cv2
import numpy as np

# Load image and template
img = cv2.imread("34343.jpg")
template = cv2.imread("54545.jpg")

# Resize image only
img = cv2.resize(img, (400, 500))
template = cv2.resize(template, (400, 500))

# Convert to grayscale
img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)

# Blur for stability
img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
template_gray = cv2.GaussianBlur(template_gray, (5, 5), 0)

best_val = -1
best_loc = None
best_size = None

# Multi-scale loop
for scale in np.linspace(0.3, 1.2, 30):
    resized = cv2.resize(
        template_gray,
        None,
        fx=scale,
        fy=scale,
        interpolation=cv2.INTER_AREA
    )

    h, w = resized.shape[:2]

    if h > img_gray.shape[0] or w > img_gray.shape[1]:
        continue

    result = cv2.matchTemplate(
        img_gray,
        resized,
        cv2.TM_CCOEFF_NORMED
    )

    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)

    if max_val > best_val:
        best_val = max_val
        best_loc = max_loc
        best_size = (w, h)

# Draw result if confidence is acceptable
if best_val > 0.5:
    x, y = best_loc
    w, h = best_size
    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
    print(f"Match score: {best_val:.2f}")
else:
    print("Intet brugbart match")

cv2.imshow("Result", img)
cv2.waitKey(0)
cv2.destroyAllWindows()
