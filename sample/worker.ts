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
        this.model = AutoModelForSeq2SeqLM.fromPretrained(modelId, modelsPath, async progress => {
            const message = `Loading model... ${progress*100}%`;
            postMessage({"inputText": "", "outputText": message, "complete": false});
        });
    }
}

interface Seq2SeqCommand {
    inputText: string;
    maxLength: number;
    topK: number | undefined;
    modelId: string;
    modelsPath: string;
}

function commandsEqual(req: Seq2SeqCommand, other: Seq2SeqCommand) {
    return req.inputText === other.inputText &&
        req.modelId === other.modelId &&
        req.modelsPath === other.modelsPath &&
        req.maxLength === other.maxLength &&
        req.topK === other.topK;
}

let lastRequest: Seq2SeqCommand | null = null;
let executingRequest: Seq2SeqCommand | null = null;

let executingModelData: ModelData | null = null;

function getModelData(modelId: string, modelsPath: string) {
    if (executingModelData !== null && executingModelData.modelId === modelId && executingModelData.modelsPath === modelsPath) {
        return executingModelData;
    }
    else {
        return new ModelData(modelId, modelsPath);
    }
}

const generate = async (request: Seq2SeqCommand) => {
    const model = getModelData(request.modelId, request.modelsPath);
    executingModelData = model;
    executingRequest = request;
    function shouldContinue(): boolean {
        return executingRequest !== null && commandsEqual(request, executingRequest);
    }
    // console.log(`Generating from input '${request.inputText}' with model '${request.modelId}'`);
    const inputText = request.inputText;
    const inputTokenIds = await model.tokenizer.encode(inputText);
    // console.log(`Generating from tokens: ${inputTokenIds}`);
    const generationOptions = {
        "maxLength": request.maxLength,
        "topK": request.topK,
    };
    function delayMillis(millis: number) {
        return new Promise((resolve) => {
            setTimeout(() => {
            resolve(0);
            }, millis);
        });
    }
    async function generateProgress(outputTokenIds: number[], forInputIds: number[]) {
        if (shouldContinue()) {
            const outputText = (await model.tokenizer.decode(outputTokenIds, true)).trim();
            postMessage({"inputText": inputText, "outputText": outputText, "complete": false});
            await delayMillis(1);
            return true;
        }
        else {
            // console.log("Generation cancelled");
            return false;
        }
    }
    const finalOutputTokenIds = await model.model.generate(inputTokenIds, generationOptions, generateProgress);
    const finalOutput = (await model.tokenizer.decode(finalOutputTokenIds, true)).trim();
    postMessage({"inputText": inputText, "outputText": finalOutput, "complete": true});
}

// console.log("Hello from worker.ts");
// translate("Hello world", "English", "French");

onmessage = (e) => {
    const request = e.data;
    lastRequest = request;
    // console.log('Message received from main script:', request);
    generate(request);    
}
