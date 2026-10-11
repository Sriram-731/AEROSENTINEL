import os
import math
from PIL import Image, ImageDraw, ImageFont

def generate_hero_image(filepath="assets/hero_drone_vision.png", width=960, height=540):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    img = Image.new("RGBA", (width, height), (8, 13, 22, 255))
    draw = ImageDraw.Draw(img)

    # 1. Background gradient and terrain lines
    for y in range(height):
        ratio = y / height
        r = int(8 + (16 - 8) * ratio)
        g = int(13 + (24 - 13) * ratio)
        b = int(22 + (39 - 22) * ratio)
        draw.line([(0, y), (width, y)], fill=(r, g, b, 255))

    # Perspective 3D grid simulating terrain beneath drone
    horizon_y = int(height * 0.42)
    center_x = width // 2

    # Horizontal grid lines with perspective spacing
    for i in range(1, 15):
        y_pos = horizon_y + int((height - horizon_y) * (math.pow(i / 14, 2.2)))
        alpha = int(30 + 100 * (i / 14))
        draw.line([(0, y_pos), (width, y_pos)], fill=(0, 229, 255, alpha), width=1)

    # Vanishing perspective lines radiating from horizon
    for x_offset in range(-width, width * 2, 80):
        start_pt = (center_x + int((x_offset - center_x) * 0.08), horizon_y)
        end_pt = (x_offset, height)
        draw.line([start_pt, end_pt], fill=(0, 229, 255, 45), width=1)

    # Subtle mountain/terrain contour silhouettes in distance
    points = [(0, horizon_y)]
    for x in range(0, width + 40, 40):
        y_variation = horizon_y - int(30 * math.sin(x * 0.015) + 18 * math.cos(x * 0.035))
        points.append((x, y_variation))
    points.append((width, horizon_y))
    draw.polygon(points, fill=(16, 24, 39, 255), outline=(59, 130, 246, 120))

    # 2. Aerospace HUD reticles & artificial horizon
    hud_cyan = (0, 229, 255, 230)
    hud_blue = (59, 130, 246, 180)
    hud_dim = (0, 229, 255, 80)

    # Center target reticle
    r_outer = 64
    draw.ellipse([center_x - r_outer, horizon_y - r_outer, center_x + r_outer, horizon_y + r_outer], outline=hud_dim, width=1)
    r_inner = 32
    draw.ellipse([center_x - r_inner, horizon_y - r_inner, center_x + r_inner, horizon_y + r_inner], outline=hud_cyan, width=2)
    # Center crosshair
    draw.line([(center_x - 16, horizon_y), (center_x + 16, horizon_y)], fill=hud_cyan, width=2)
    draw.line([(center_x, horizon_y - 16), (center_x, horizon_y + 16)], fill=hud_cyan, width=2)

    # Artificial pitch ladder lines
    for p_offset in [-50, -25, 25, 50]:
        y_p = horizon_y + p_offset
        w_ladder = 36 if abs(p_offset) == 25 else 22
        draw.line([(center_x - w_ladder - 40, y_p), (center_x - 40, y_p)], fill=hud_blue, width=2)
        draw.line([(center_x + 40, y_p), (center_x + w_ladder + 40, y_p)], fill=hud_blue, width=2)

    # 3. Computer Vision Simulated Bounding Boxes (Illustration context only)
    # Bounding Box 1: Obstacle Tower / Pylon
    box1 = [180, horizon_y - 80, 290, horizon_y + 90]
    draw.rectangle(box1, outline=hud_cyan, width=2)
    # Corner brackets for tech feel
    c_len = 12
    draw.line([(box1[0], box1[1]), (box1[0] + c_len, box1[1])], fill=(255, 255, 255, 255), width=3)
    draw.line([(box1[0], box1[1]), (box1[0], box1[1] + c_len)], fill=(255, 255, 255, 255), width=3)
    draw.rectangle([box1[0], box1[1] - 22, box1[0] + 120, box1[1]], fill=(0, 229, 255, 200))

    # Bounding Box 2: Aerial structure
    box2 = [640, horizon_y - 30, 780, horizon_y + 120]
    draw.rectangle(box2, outline=(59, 130, 246, 220), width=2)
    draw.rectangle([box2[0], box2[1] - 22, box2[0] + 140, box2[1]], fill=(59, 130, 246, 200))

    # 4. HUD Telemetry overlay texts and brackets
    # Outer HUD frame corners
    margin = 28
    f_len = 40
    # Top-left
    draw.line([(margin, margin), (margin + f_len, margin)], fill=hud_cyan, width=2)
    draw.line([(margin, margin), (margin, margin + f_len)], fill=hud_cyan, width=2)
    # Top-right
    draw.line([(width - margin, margin), (width - margin - f_len, margin)], fill=hud_cyan, width=2)
    draw.line([(width - margin, margin), (width - margin, margin + f_len)], fill=hud_cyan, width=2)
    # Bottom-left
    draw.line([(margin, height - margin), (margin + f_len, height - margin)], fill=hud_cyan, width=2)
    draw.line([(margin, height - margin), (margin, height - margin - f_len)], fill=hud_cyan, width=2)
    # Bottom-right
    draw.line([(width - margin, height - margin), (width - margin - f_len, height - margin)], fill=hud_cyan, width=2)
    draw.line([(width - margin, height - margin), (width - margin, height - margin - f_len)], fill=hud_cyan, width=2)

    img.save(filepath, "PNG")
    print(f"Generated {filepath} ({width}x{height})")

if __name__ == "__main__":
    generate_hero_image()
