Este projeto converte números entre o sistema número tradicional e o sistema cisterciense utilizando OpenCV e uma interface frontend com Node.js, HTML e CSS.  
A aplicação consegue gerar a imagem de um número cisterciense a partir de um número tracicional (1,2,3 etc.), quanto reconhecer e interpretar uma imagem de um número cisterciense.

---

## Funcionalidades

### Conversão Tradicional ➜ Cisterciense
- **Entrada**: Número entre 1 e 9999
- **Saída**: Imagem em PNG do número representado no sistema cisterciense

### Conversão Cisterciense ➜ Arábico
- **Entrada**: Imagem carregada pelo usuário
- **Saída**: Número arábico correspondente

---

## Tecnologias

### Backend
  - Python 3.12+
  - Flask (API)
  - OpenCV (processamento de imagem)
  - scikit-image (SSIM para similaridade)
  - Pillow

### Frontend
  - Node.js 18+
  - HTML, CSS, JS puro
---

## Como Rodar
  - Instale as dependências encontradas no arquivo ./python/requirements.txt

```plaintext
     pip install -r python/requirements.txt
```
  - Instale o NPM

```plaintext
    npm install
```



---


## 📁 Estrutura do Projeto

```plaintext
AlgoritmoCistercinese/
├── dataset-numbers/              # Imagens para reconhecimento das imagens
├── number-images/                # Imagens para gerar o numero inserido pelo usuário em tela
├── output/                       # Imagem gerada do numero inserido
├── public/
│   └── index.html                # Interface com frontend
│   └── style.css                 # Style do frontend
├── python/
│   └── recognize_number.py       # Verificação da imagem inserida pelo usuário
│   └── requirements.txt          # Dependências do projeto 
├── uploads/                      # Arquivo temporário da imagens inserirdas
├── utils/
│   └── imageComposer.js          # Gerador da imagem referente ao número digitado
├── app.js                        # Arquivo princial da aplicação node





