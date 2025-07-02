import os
import sys
import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim

# Threshold mínimo para considerar match válido
MIN_CONFIDENCE = 0.6

def prepare_image(path):
    """
    Lê a imagem (com canal alpha se houver), remove fundo transparente pintando de branco,
    converte para grayscale, binariza com Otsu e recorta para o conteúdo.
    """
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"Não foi possível carregar a imagem: {path}")

    # Verifica se tem canal alpha
    if img.shape[2] == 4:
        # Split channels
        b, g, r, a = cv2.split(img)
        alpha = a / 255.0

        # Composita sobre fundo branco
        white_bg = np.ones_like(b, dtype=np.uint8) * 255
        b = (b * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        g = (g * alpha + white_bg * (1 - alpha)).astype(np.uint8)
        r = (r * alpha + white_bg * (1 - alpha)).astype(np.uint8)

        img = cv2.merge([b, g, r])

    # Converte para grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Binariza com Otsu
    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Corrige inversão de cores se necessário
    if np.mean(binary) < 127:
        binary = cv2.bitwise_not(binary)

    # Recorta para o conteúdo real (remove bordas extras)
    binary = crop_to_content(binary)

    # Redimensiona com padding para tamanho padrão
    return resize_with_padding(binary)

def crop_to_content(binary):
    """
    Recorta a imagem ao retângulo mínimo que contenha os traços.
    """
    coords = cv2.findNonZero(255 - binary)
    if coords is None:
        return binary
    x, y, w, h = cv2.boundingRect(coords)
    cropped = binary[y:y+h, x:x+w]
    return cropped

def resize_with_padding(img, target_size=(200, 200)):
    """
    Redimensiona mantendo aspecto e adiciona padding branco.
    """
    h, w = img.shape
    scale = min(target_size[0] / w, target_size[1] / h)
    new_w, new_h = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_AREA)
    canvas = np.ones(target_size, dtype=np.uint8) * 255
    x_offset = (target_size[0] - new_w) // 2
    y_offset = (target_size[1] - new_h) // 2
    canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized
    return canvas

def dice_coeff(a, b):
    """
    Calcula Dice coefficient entre duas imagens binárias.
    """
    a = a < 128
    b = b < 128
    intersection = np.logical_and(a, b).sum()
    if a.sum() + b.sum() == 0:
        return 1.0
    return 2.0 * intersection / (a.sum() + b.sum())

def compare_images(user_img, candidate_img, method='dice'):
    if method == 'ssim':
        return ssim(user_img, candidate_img, data_range=255)
    elif method == 'dice':
        return dice_coeff(user_img, candidate_img)
    else:
        raise ValueError(f"Método desconhecido: {method}")

def find_best_match(user_image_path, dataset_folder, method='dice'):
    """
    Percorre o dataset procurando o número com maior similaridade.
    """
    user_img = prepare_image(user_image_path)
    best_match = (None, -1)

    dataset_files = sorted(
        [f for f in os.listdir(dataset_folder) if f.endswith('.png')],
        key=lambda x: int(x.replace('.png', ''))
    )
    total = len(dataset_files)

    print(f"Iniciando comparação com {total} imagens do dataset...\n", file=sys.stderr)

    for idx, filename in enumerate(dataset_files, 1):
        number = filename.replace('.png', '')
        candidate_path = os.path.join(dataset_folder, filename)

        try:
            candidate_img = prepare_image(candidate_path)
            score = compare_images(user_img, candidate_img, method=method)

            print(f"[{idx}/{total}] Comparando com {number.zfill(4)}.png → Similaridade: {score:.4f}", file=sys.stderr)

            if score > best_match[1]:
                best_match = (number, score)

        except Exception as e:
            print(f"[{idx}/{total}] Erro ao comparar {filename}: {e}", file=sys.stderr)

    print("\nComparação finalizada.", file=sys.stderr)

    if best_match[1] < MIN_CONFIDENCE:
        print(f"⚠️ Nenhum match com confiança suficiente (score {best_match[1]:.4f} abaixo de {MIN_CONFIDENCE})", file=sys.stderr)
        return None, best_match[1]

    return best_match

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Uso: python recognize_number.py <imagem_usuario> <pasta_dataset>")
        sys.exit(1)

    user_image = sys.argv[1]
    dataset_folder = sys.argv[2]

    # Escolha o método: 'ssim' ou 'dice'
    method = 'dice'

    best_number, best_score = find_best_match(user_image, dataset_folder, method=method)

    if best_number is None:
        print(f"None|{best_score:.4f}")
    else:
        print(f"{best_number}|{best_score:.4f}")
