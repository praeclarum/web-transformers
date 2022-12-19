import { AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration } from '../lib';

// This is the command message sent from the UI to the worker
interface Seq2SeqCommand {
    inputText: string;
    maxLength: number;
    topK: number | undefined;
    modelId: string;
    modelsPath: string;
}

class ModelData {
    modelId: string;
    modelsPath: string;
    tokenizer: AutoTokenizer;
    model: AutoModelForSeq2SeqLM;
    constructor(modelId: string, modelsPath: string) {
        this.modelId = modelId;
        this.modelsPath = modelsPath;
        this.tokenizer = AutoTokenizer.fromPretrained(modelId, modelsPath);
        this.model = new T5ForConditionalGeneration(modelId, modelsPath, async progress => {
            const message = `Loading model... ${progress*100}%`;
            postMessage({"inputText": "", "outputText": message, "complete": false});
        });
    }
}

function commandsEqual(req: Seq2SeqCommand, other: Seq2SeqCommand) {
    return req.inputText === other.inputText &&
        req.modelId === other.modelId &&
        req.modelsPath === other.modelsPath &&
        req.maxLength === other.maxLength &&
        req.topK === other.topK;
}

let executingCommand: Seq2SeqCommand | null = null;

let executingModelData: ModelData | null = null;

function getModelData(modelId: string, modelsPath: string) {
    if (executingModelData !== null && executingModelData.modelId === modelId && executingModelData.modelsPath === modelsPath) {
        return executingModelData;
    }
    else {
        return new ModelData(modelId, modelsPath);
    }
}

const generate = async (command: Seq2SeqCommand) => {
    const model = getModelData(command.modelId, command.modelsPath);
    executingModelData = model;
    executingCommand = command;
    function shouldContinue(): boolean {
        return executingCommand !== null && commandsEqual(command, executingCommand);
    }
    // console.log(`Generating from input '${command.inputText}' with model '${command.modelId}'`);
    const inputText = command.inputText;
    const inputTokenIds = await model.tokenizer.encode(inputText);
    // console.log(`Generating from tokens: ${inputTokenIds}`);
    const generationOptions = {
        "maxLength": command.maxLength,
        "topK": command.topK,
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

onmessage = (e: MessageEvent<Seq2SeqCommand>) => {
    const command = e.data;
    // console.log('Message received from main script:', command);
    generate(command);
}
