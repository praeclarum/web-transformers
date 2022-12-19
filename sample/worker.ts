import { AutoTokenizer, AutoModelForSeq2SeqLM } from '../lib';

class ModelData {
    modelId: string;
    modelsPath: string;
    tokenizer: AutoTokenizer;
    model: AutoModelForSeq2SeqLM;
    constructor(modelId: string, modelsPath: string) {
        this.modelId = modelId;
        this.modelsPath = modelsPath;
        this.tokenizer = AutoTokenizer.fromPretrained(modelId, modelsPath);
        this.model = AutoModelForSeq2SeqLM.fromPretrained(modelId, modelsPath);
    }
}

class GenerationRequest {
    input: string = "";
    maxLength: number = 50;
    topK: number | undefined;
    modelId: string;
    modelsPath: string;
    constructor(modelId: string, modelsPath: string) {
        this.modelId = modelId;
        this.modelsPath = modelsPath;
    }
    equals(other: GenerationRequest) {
        return this.input === other.input &&
            this.modelId === other.modelId &&
            this.modelsPath === other.modelsPath &&
            this.maxLength === other.maxLength &&
            this.topK === other.topK;
    }
}

let lastRequest: GenerationRequest | null = null;
let executingRequest: GenerationRequest | null = null;

let executingModelData: ModelData | null = null;

const generate = async (request: GenerationRequest) => {
    const model = new ModelData(request.modelId, request.modelsPath);
    executingModelData = model;
    function shouldContinue(): boolean {
        return executingRequest !== null && request.equals(executingRequest);
    }
    console.log(`Generating from input '{request.input}' with model '{request.modelId}'`);
    const inputText = request.input;
    const inputTokenIds = await model.tokenizer.encode(inputText);
    console.log(`Generating from tokens: {inputTokenIds}`);
    const generationOptions = {
        "maxLength": request.maxLength,
        "topK": request.topK,
    };
    async function generateProgress(outputTokenIds: number[], forInputIds: number[]) {
        if (shouldContinue()) {
            const outputText = (await model.tokenizer.decode(outputTokenIds, true)).trim();
            postMessage({"inputText": inputText, "outputText": outputText, "complete": false});
            return true;
        }
        else {
            return false;
        }
    }
    const finalOutputTokenIds = await model.model.generate(inputTokenIds, generationOptions, generateProgress);
    const finalOutput = (await model.tokenizer.decode(finalOutputTokenIds, true)).trim();
    postMessage({"inputText": inputText, "outputText": finalOutput, "complete": true});
}

console.log("Hello from worker.ts");
// translate("Hello world", "English", "French");

onmessage = (e) => {
    const request = e.data;
    lastRequest = request;
    console.log('Message received from main script:', request);
    generate(request);    
}
