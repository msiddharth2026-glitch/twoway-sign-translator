import os
import shutil

DATASET_DIR = 'dataset/original_images'
TEST_DIR = 'test'

os.makedirs(TEST_DIR, exist_ok=True)

letter_to_num = {
    'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7,
    'H': 8, 'I': 9, 'J': 10, 'K': 11, 'L': 12, 'M': 13, 'N': 14,
    'O': 15, 'P': 16, 'Q': 17, 'R': 18, 'S': 19, 'T': 20,
    'U': 21, 'V': 22, 'W': 23, 'X': 24, 'Y': 25, 'Z': 26
}

if os.path.exists(DATASET_DIR):
    for letter, num in letter_to_num.items():
        src_dir = os.path.join(DATASET_DIR, letter)
        if os.path.exists(src_dir):
            images = [f for f in os.listdir(src_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
            if images:
                shutil.copy(os.path.join(src_dir, images[0]), os.path.join(TEST_DIR, f'{num}.jpg'))
                print(f"Copied {images[0]} -> test/{num}.jpg")
else:
    print(f"Dataset directory '{DATASET_DIR}' not found.")

space_src = os.path.join(DATASET_DIR, 'A')
if os.path.exists(space_src):
    images = [f for f in os.listdir(space_src) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
    if images:
        shutil.copy(os.path.join(space_src, images[0]), os.path.join(TEST_DIR, 'space.jpg'))
        print("Created space.jpg (copy of A sign)")

print("Setup complete!")
