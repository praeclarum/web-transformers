import { AutoTokenizer, AutoModelForSeq2SeqLM } from '../lib';

let modelId = "t5-small";
const tokenizer = AutoTokenizer.fromPretrained(modelId, "/models");
const translate = async (inputText: string, inputLang: string, outputLang: string) => {
    console.log(`Translating {inputText} from {inputLang} to {outputLang}`);
    const generationOptions = {
        "maxLength": 50,
        "topK": 0,
    };
    const fullInput = `translate ${inputLang} to ${outputLang}: ${inputText.trim()}`;
    const inputTokenIds = await tokenizer.encode(fullInput);
    console.log(inputTokenIds);
}

console.log("Hello from worker.ts");
translate("Hello world", "English", "French");

onmessage = (e) => {
    const command = e.data;
    console.log('Message received from main script', command);
    const workerResult = command.input + "!!!";
    console.log('Posting message back to main script');
    postMessage(workerResult);
}