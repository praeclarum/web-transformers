# web-transformers

[![Build and Test](https://github.com/praeclarum/web-transformers/actions/workflows/build.yml/badge.svg?branch=main)](https://github.com/praeclarum/web-transformers/actions/workflows/build.yml)

This library enables you to run huggingface transformer models directly in the browser.
It accomplishes this by running the models using the
[ONNX Runtime JavaScript API](https://github.com/microsoft/onnxruntime/tree/main/js)
and by implementing its own JavaScript-only tokenization library.

At the moment, it is compatible with Google's T5 models, but it was designed to be expanded.
I hope to support GPT2, Roberta, and InCoder in the future.


## Usage

The following example code shows how to load a tokenizer and a transformer model.
It then uses those objects to create a `generate` functions which generates
output text from input text.

```typescript
import { AutoTokenizer, T5ForConditionalGeneration } from 'web-transformers';

// Load the tokenizer and model
const tokenizer = AutoTokenizer.fromPretrained(modelId, modelsPath);
const model = new T5ForConditionalGeneration(modelId, modelsPath);

// Generate text using Seq2Seq
async function generate(inputText: string) {
    // Tokenize the text
    const inputTokenIds = await tokenizer.encode(inputText);
    // Generate output
    const generationOptions = {
        "maxLength": 50,
        "topK": 10,
    };
    const outputTokenIds = await model.generate(inputTokenIds, generationOptions);
    // Convert output tokens into text
    const outputText = await tokenizer.decode(finalOutputTokenIds, true);
    return outputText;
};

// Translate
console.log(await generate("translate English to French: Hello World!"));
```

## Sample Web App

This repo contains a [sample nextjs app](sample) that translates text using a transformer neural network.

Before you can run the sample, you need to download the model files. You can do this by running the following command:

```bash
cd sample
npm run download_models
```

Then you can run the sample with:

```bash
npm run dev
```


## Models

Currently only the *T5* network is supported.

### Sampling

The neural network outputs the logarithm of the probability of each token.
In order to get a token, a probabilistic sample has to be taken.
The following algorithms are implemented:

* *Greedy*: Take the token with the highest probability.
* *Top-k*: Take the top-k tokens with the highest probability.


## Sponsorship

This library was made possible thanks to the financial support of [Reflect](https://reflect.app). Thanks Reflect!


## License

Copyright © 2022 [Frank A. Krueger](https://github.com/praeclarum). This project is [MIT](LICENSE.md) licensed.
