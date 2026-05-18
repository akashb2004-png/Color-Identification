import cv2
import numpy as np

mouse_x, mouse_y = -1, -1


def on_mouse(event, x, y, flags, param):
    global mouse_x, mouse_y
    mouse_x, mouse_y = x, y


def get_color_name(h, s, v):
    # --- Achromatic ---
    if v < 30:
        return "Black"
    if v < 60 and s < 50:
        return "Dark Gray"
    if s < 35:
        if v > 220:
            return "White"
        if v > 185:
            return "Light Gray"
        if v > 110:
            return "Gray"
        return "Dark Gray"

    # --- Browns / Neutrals (low saturation warm tones) ---
    if 10 <= h <= 25 and s > 60 and v < 80:
        return "Dark Brown"
    if 8 <= h <= 25 and 60 < s <= 160 and 80 <= v < 160:
        return "Brown"
    if 10 <= h <= 30 and 30 < s <= 100 and v >= 150:
        return "Beige"
    if 8 <= h <= 25 and s > 160 and 80 <= v < 160:
        return "Chocolate"

    # --- Reds (hue wraps around 0/179) ---
    red_hue = h <= 10 or h >= 165

    if red_hue:
        if s > 180 and v < 90:
            return "Maroon"
        if s > 150 and v > 160:
            return "Red"
        if s > 100 and v > 100:
            return "Dark Red"
        if s <= 150 and v > 160:
            return "Pink"
        if s <= 100 and v > 140:
            return "Light Pink"
        return "Red"

    # --- Orange-Reds ---
    if 10 < h <= 14:
        if s > 180 and v > 150:
            return "Orange Red"
        if s > 100 and v > 120:
            return "Coral"
        if 80 < s <= 180 and v > 150:
            return "Salmon"

    # --- Oranges ---
    if 14 < h <= 25:
        if s > 200 and v > 180:
            return "Orange"
        if s > 150 and v > 140:
            return "Dark Orange"
        if 80 < s <= 200 and v > 160:
            return "Peach"

    # --- Yellows ---
    if 25 < h <= 33:
        if s > 180 and v > 180:
            return "Yellow"
        if s > 120 and 100 <= v <= 200:
            return "Gold"
        if s <= 120 and v > 160:
            return "Cream"

    # --- Yellow-Greens / Olive ---
    if 33 < h <= 48:
        if v < 120:
            return "Olive"
        if s > 150:
            return "Yellow Green"
        return "Light Olive"

    # --- Greens ---
    if 48 < h <= 85:
        if s > 180 and v > 180:
            return "Lime Green"
        if v < 80:
            return "Dark Green"
        if v < 140 and s > 100:
            return "Forest Green"
        if s < 100 and v > 160:
            return "Mint"
        return "Green"

    # --- Cyans / Teals ---
    if 85 < h <= 100:
        if v < 120:
            return "Teal"
        if s > 150:
            return "Cyan"
        return "Light Cyan"

    # --- Blues ---
    if 100 < h <= 115:
        if s < 120 and v > 180:
            return "Sky Blue"
        if v < 100:
            return "Navy Blue"
        if s > 180:
            return "Azure"
        return "Light Blue"

    if 115 < h <= 130:
        if v < 100:
            return "Dark Blue"
        if s > 180 and v > 150:
            return "Blue"
        if s > 120:
            return "Royal Blue"
        return "Steel Blue"

    # --- Indigos / Purples ---
    if 130 < h <= 145:
        if v < 100:
            return "Dark Indigo"
        return "Indigo"

    if 145 < h <= 158:
        if s < 120 and v > 160:
            return "Lavender"
        if v < 100:
            return "Dark Purple"
        return "Purple"

    # --- Magentas / Violets ---
    if 158 < h <= 168:
        if s > 180 and v > 160:
            return "Magenta"
        if s > 100 and v > 140:
            return "Violet"
        if s < 120:
            return "Lavender"
        return "Purple"

    if 168 < h <= 179:
        if s > 180 and v > 180:
            return "Hot Pink"
        if s > 100 and v > 150:
            return "Deep Pink"
        return "Pink"

    return "Unknown"


def hsv_to_bgr(h, s, v):
    pixel = np.uint8([[[h, s, v]]])
    bgr = cv2.cvtColor(pixel, cv2.COLOR_HSV2BGR)
    b, g, r = int(bgr[0][0][0]), int(bgr[0][0][1]), int(bgr[0][0][2])
    return b, g, r


def draw_overlay(frame, cx, cy, color_name, hex_str, r, g, b, swatch_bgr):
    # Semi-transparent background panel
    panel = frame.copy()
    cv2.rectangle(panel, (10, 10), (310, 130), (20, 20, 20), -1)
    cv2.addWeighted(panel, 0.65, frame, 0.35, 0, frame)

    # Color swatch
    sw_b, sw_g, sw_r = swatch_bgr
    cv2.rectangle(frame, (22, 22), (90, 90), (sw_b, sw_g, sw_r), -1)
    cv2.rectangle(frame, (22, 22), (90, 90), (220, 220, 220), 1)

    # Color name
    cv2.putText(frame, color_name, (100, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 0.85, (255, 255, 255), 2, cv2.LINE_AA)
    # Hex
    cv2.putText(frame, hex_str, (100, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (180, 180, 180), 1, cv2.LINE_AA)
    # RGB
    cv2.putText(frame, f"RGB  {r}, {g}, {b}", (100, 102),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (150, 150, 150), 1, cv2.LINE_AA)
    # Hint
    cv2.putText(frame, "Move mouse to sample  |  Q to quit", (12, 125),
                cv2.FONT_HERSHEY_SIMPLEX, 0.38, (120, 120, 120), 1, cv2.LINE_AA)

    # Crosshair at sample point
    cv2.line(frame, (cx - 14, cy), (cx - 4, cy), (0, 0, 0), 3)
    cv2.line(frame, (cx + 4, cy), (cx + 14, cy), (0, 0, 0), 3)
    cv2.line(frame, (cx, cy - 14), (cx, cy - 4), (0, 0, 0), 3)
    cv2.line(frame, (cx, cy + 4), (cx, cy + 14), (0, 0, 0), 3)
    cv2.line(frame, (cx - 14, cy), (cx - 4, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx + 4, cy), (cx + 14, cy), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy - 14), (cx, cy - 4), (255, 255, 255), 1)
    cv2.line(frame, (cx, cy + 4), (cx, cy + 14), (255, 255, 255), 1)

    # Small dot at exact sample point
    cv2.circle(frame, (cx, cy), 3, (sw_b, sw_g, sw_r), -1)
    cv2.circle(frame, (cx, cy), 3, (255, 255, 255), 1)


def main():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    win = "Real-Time Color Identifier"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)

    sample_radius = 6

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to read frame.")
            break

        frame = cv2.flip(frame, 1)
        h_frame, w_frame = frame.shape[:2]

        cx = mouse_x if 0 <= mouse_x < w_frame else w_frame // 2
        cy = mouse_y if 0 <= mouse_y < h_frame else h_frame // 2

        # Sample region around cursor
        y1, y2 = max(0, cy - sample_radius), min(h_frame, cy + sample_radius)
        x1, x2 = max(0, cx - sample_radius), min(w_frame, cx + sample_radius)
        roi = frame[y1:y2, x1:x2]

        if roi.size == 0:
            continue

        hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h_med = int(np.median(hsv_roi[:, :, 0]))
        s_med = int(np.median(hsv_roi[:, :, 1]))
        v_med = int(np.median(hsv_roi[:, :, 2]))

        color_name = get_color_name(h_med, s_med, v_med)
        swatch_b, swatch_g, swatch_r = hsv_to_bgr(h_med, s_med, v_med)

        # Use actual sampled pixel color for hex/RGB display
        bgr_roi = roi.reshape(-1, 3)
        bgr_median = np.median(bgr_roi, axis=0).astype(int)
        b_val, g_val, r_val = int(bgr_median[0]), int(bgr_median[1]), int(bgr_median[2])
        hex_str = f"#{r_val:02X}{g_val:02X}{b_val:02X}"

        draw_overlay(frame, cx, cy, color_name, hex_str,
                     r_val, g_val, b_val, (swatch_b, swatch_g, swatch_r))

        cv2.imshow(win, frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
