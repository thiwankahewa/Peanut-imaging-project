import cv2

IMG_PATH = "reference.png" 
img = cv2.imread(IMG_PATH, cv2.IMREAD_UNCHANGED)

if img is None:
    raise FileNotFoundError(f"Cannot read image: {IMG_PATH}")

h, w = img.shape[:2]

# --- Choose max window size (tune if needed) ---
MAX_W = 1000
MAX_H = 700

# Compute scale so the image fits inside MAX_W x MAX_H
scale = min(MAX_W / w, MAX_H / h, 1.0)   # never upscale, only downscale
disp_w = int(w * scale)
disp_h = int(h * scale)

# Create the display image
if scale < 1.0:
    disp_img = cv2.resize(img, (disp_w, disp_h), interpolation=cv2.INTER_AREA)
else:
    disp_img = img.copy()


def mouse_event(event, x_disp, y_disp, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        # Map display coords back to original coords
        x = int(x_disp / scale)
        y = int(y_disp / scale)
        # Safety clamp
        x = min(max(x, 0), w - 1)
        y = min(max(y, 0), h - 1)

        pixel_val = img[y, x]
        print(f"x={x}, y={y}, pixel={pixel_val}")

cv2.namedWindow("image", cv2.WINDOW_NORMAL)
cv2.setMouseCallback("image", mouse_event)

while True:
    cv2.imshow("image", disp_img)
    key = cv2.waitKey(1) & 0xFF
    if key == 27:   # ESC
        break

cv2.destroyAllWindows()
