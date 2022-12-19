# web-transformers

[![Build and test](https://github.com/praeclarum/web-transformers/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/praeclarum/web-transformers/actions/workflows/build.yml)

Transformer neural networks in the browser.


## Usage

```typescript
// Load the tokenizer and model
const tokenizer = AutoTokenizer.fromPretrained(modelId, modelsPath);
const model = AutoModelForSeq2SeqLM.fromPretrained(modelId, modelsPath, async function (progress) {
    console.log(`Loading the neural network... ${Math.round(progress * 100)}%`);
});

// Generate text using Seq2Seq
async function generate(inputText: string) {
    const generationOptions = {
        "maxLength": 50,
        "topK": 0,
    };
    const inputTokenIds = await tokenizer.encode(inputText);
    async function generateProgress(outputTokenIds: number[], forInputIds: number[]) {
        let shouldContinue = true;
        return shouldContinue;
    }
    const finalOutputTokenIds = await model.generate(inputTokenIds, generationOptions, generateProgress);
    const finalOutput = (await tokenizer.decode(finalOutputTokenIds, true)).trim();
    console.log(finalOutput);
};
```

## Sample Web App

This repo contains a sample nextjs app that translates text using a transformer neural network.

Before you can run the sample, you need to download the model files. You can do this by running the following command:

```bash
cd sample
npm run download_models
```

Then you can run the sample with:

```bash
npm run dev
```

## License

Copyright © 2022 [Frank A. Krueger](https://github.com/praeclarum).<br />
This project is [MIT](LICENSE.md) licensed.
