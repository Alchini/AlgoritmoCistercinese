const sharp = require('sharp');
const path = require('path');
const fs = require('fs');

async function composeNumberImage(numbers, imgDir, outputPath) {
  const images = await Promise.all(numbers.map(async n => {
    const file = path.join(imgDir, `${n}.png`);
    if (!fs.existsSync(file)) throw new Error(`Imagem não encontrada: ${n}.png`);
    return sharp(file).ensureAlpha();
  }));

  const metadata = await Promise.all(images.map(img => img.metadata()));
  const width = Math.max(...metadata.map(m => m.width));
  const height = Math.max(...metadata.map(m => m.height));

  const composite = await sharp({
    create: {
      width: width,
      height: height,
      channels: 4,
      background: { r: 0, g: 0, b: 0, alpha: 0 } // fundo transparente
    }
  }).composite(
    await Promise.all(images.map(async img => {
      const buffer = await img.resize(width, height).toBuffer(); // redimensionar para alinhar
      return { input: buffer, top: 0, left: 0 };
    }))
  ).png().toFile(outputPath);
}

module.exports = { composeNumberImage };
