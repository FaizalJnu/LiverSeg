import os
from PIL import Image

RESULT_DIR = "outputs/test_results"
OUT_PATH = "outputs/sample_results.png"

# Exact files you want, in the order you want them
selected_files = [
    "Patient 17_pat17_image_27.png",
    "Patient 18_pat18_image_27.png",
    "Patient 19_pat19_image_50.png",
    "Patient 20_pat20_image_37.png",
]

def main():
    images = []

    # Load each selected file
    for fname in selected_files:
        path = os.path.join(RESULT_DIR, fname)

        if not os.path.exists(path):
            print(f"Missing file: {path}")
            continue

        img = Image.open(path)
        images.append(img)
        print("Using:", path)

    if not images:
        print("No images loaded, nothing to stack.")
        return

    # Assume all same size
    w, h = images[0].size
    total_h = h * len(images)

    # Create tall stacked image
    stack = Image.new("RGB", (w, total_h))

    y = 0
    for im in images:
        stack.paste(im, (0, y))
        y += h

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    stack.save(OUT_PATH)
    print("\nFinal sample image saved →", OUT_PATH)

if __name__ == "__main__":
    main()


