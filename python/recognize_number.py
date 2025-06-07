import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim
import time 

def prepare_image(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
    _, binary = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)
    return cv2.resize(binary, (200, 200))

def find_best_match(user_image_path, dataset_folder):
    user_img = prepare_image(user_image_path)
    best_match = (None, -1)

    dataset_files = sorted(os.listdir(dataset_folder), key=lambda x: int(x.replace('.png', '')))
    total = len(dataset_files)

    print(f"Iniciando comparação com {total} imagens do dataset...\n", file=sys.stderr)

    for idx, filename in enumerate(dataset_files, 1):
        if not filename.endswith('.png'):
            continue

        number = filename.replace('.png', '')
        candidate_path = os.path.join(dataset_folder, filename)

        try:
            candidate_img = prepare_image(candidate_path)
            score = ssim(user_img, candidate_img)

            # Log linha a linha com progresso
            print(f"[{idx}/{total}] Comparando com {number.zfill(4)}.png → Similaridade: {score:.4f}", file=sys.stderr)

            if score > best_match[1]:
                best_match = (number, score)

            # time.sleep(0.01)  # opcional: pausa para facilitar leitura visual

        except Exception as e:
            print(f"[{idx}/{total}] Erro ao comparar {filename}: {e}", file=sys.stderr)

    print("\nComparação finalizada.", file=sys.stderr)
    return best_match

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python recognize_number.py <imagem_usuario> <pasta_dataset>")
        sys.exit(1)

    user_image = sys.argv[1]
    dataset_folder = sys.argv[2]

    best_number, best_score = find_best_match(user_image, dataset_folder)
    print(f"{best_number}|{best_score:.4f}")
