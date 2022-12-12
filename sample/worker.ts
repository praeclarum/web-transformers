import { AutoTokenizer, AutoModelForSeq2SeqLM } from '../lib';

let requestedInputText = "";
let requestedModelId = "t5-small";
let requestedModelsPath = "/models";

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

const generate = async () => {
    const inputText = requestedInputText;
    const model = new ModelData(requestedModelId, requestedModelsPath);
    console.log(`Translating {inputText} from {inputLang} to {outputLang}`);
    const generationOptions = {
        "maxLength": 50,
        "topK": 0,
    };
    const inputTokenIds = await model.tokenizer.encode(inputText);
    console.log(inputTokenIds);
    async function generateProgress(outputTokenIds: number[], forInputIds: number[]) {
        let shouldContinue = true;
        return shouldContinue;
    }
    const finalOutputTokenIds = await model.model.generate(inputTokenIds, generationOptions, generateProgress);
    const finalOutput = (await model.tokenizer.decode(finalOutputTokenIds, true)).trim();
    postMessage(finalOutput);
}

console.log("Hello from worker.ts");
// translate("Hello world", "English", "French");

onmessage = (e) => {
    const command = e.data;
    console.log('Message received from main script', command);
    const workerResult = command.input + "!!!";
    requestedInputText = command.input;
    console.log('Posting message back to main script');
    generate();    
}
