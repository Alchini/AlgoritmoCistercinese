const express = require('express');
const path = require('path');
const fs = require('fs');
const multer = require('multer');
const { exec } = require('child_process');
const { composeNumberImage } = require('./utils/imageComposer');

const app = express();
const PORT = 3000;

const upload = multer({ dest: path.join(__dirname, 'uploads') });
const { spawn } = require('child_process');

app.use(express.static('public'));
app.use('/output', express.static(path.join(__dirname, 'output')));
app.use(express.urlencoded({ extended: true }));
app.use(express.json());

app.post('/generate', async (req, res) => {
  const { number } = req.body;
  const num = parseInt(number, 10);

  if (isNaN(num) || num < 1 || num > 9999) {
    return res.status(400).send('Número inválido. Digite entre 1 e 9999.');
  }

  const digits = decomposeNumber(num);
  const filename = `${num}.png`;
  const outputPath = path.join(__dirname, 'output', filename);

  try {
    await composeNumberImage(digits, path.join(__dirname, 'number-images'), outputPath);
    res.json({ image: `/output/${filename}` });
  } catch (err) {
    console.error(err);
    res.status(500).send('Erro ao gerar a imagem.');
  }
});

app.post('/upload', upload.single('image'), async (req, res) => {
    try {
      const imagePath = req.file.path;
      const scriptPath = path.join(__dirname, 'python', 'recognize_number.py');
      const datasetPath = path.join(__dirname, 'dataset-numbers');
  
      const pythonProcess = spawn('python', [scriptPath, imagePath, datasetPath]);
  
      let result = '';
      let errorLog = '';
  
      pythonProcess.stdout.on('data', (data) => {
        result += data.toString();
      });
  
      pythonProcess.stderr.on('data', (data) => {
        // Exibe os logs linha a linha no terminal do Node
        process.stdout.write(data.toString());
        errorLog += data.toString();
      });
  
      pythonProcess.on('close', (code) => {
        if (code !== 0) {
          console.error('Erro no script Python:', errorLog);
          return res.status(500).send('Erro ao processar a imagem.');
        }
  
        const [number, score] = result.trim().split('|');
  
        res.json({
          number: Number(number),
          score: parseFloat(score),
          message: `A imagem inserida representa o número ${number} (similaridade: ${score}).`
        });
      });
  
    } catch (err) {
      console.error(err);
      res.status(500).send('Erro ao processar a imagem.');
    }
  });

function decomposeNumber(n) {
  const parts = [];
  const units = [1000, 100, 10, 1];
  for (const u of units) {
    const val = Math.floor(n / u) * u;
    if (val > 0) parts.push(val);
    n = n % u;
  }
  return parts;
}

app.listen(PORT, () => {
  console.log(`Servidor rodando em http://localhost:${PORT}`);
});
