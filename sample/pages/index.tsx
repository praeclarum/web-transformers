import Head from 'next/head'
import Image from 'next/image'
import styles from '../styles/Home.module.css'

import { AutoTokenizer, AutoModelForSeq2SeqLM } from '../../lib';

import { useState, useRef, useCallback, FormEvent, ChangeEvent, useEffect, useMemo } from 'react';

export default function Home() {

  const modelId = "t5-small";
  const modelsPath = "/models";

  const [status, setStatus] = useState('Loading...');
  const [inputLang, setInputLang] = useState('English');
  const [outputLang, setOutputLang] = useState('French');
  const [inputText, setInputText] = useState('The universe is a dark forest.');
  const [inputDebug, setInputDebug] = useState('');
  const [outputText, setOutputText] = useState('');

  const tokenizer = useRef(AutoTokenizer.fromPretrained(modelId, modelsPath));
  const model = useRef(AutoModelForSeq2SeqLM.fromPretrained(modelId, modelsPath, async function (progress) {
    const message = `Loading the neural network... ${Math.round(progress * 100)}%`;
    setStatus(message);
    setOutputText(message);
  }));

  async function onInputChanged(event: FormEvent<HTMLTextAreaElement>) {
    const newInput = event.currentTarget.value;
    setInputText(newInput);
    translate(newInput, outputLang);
  }

  function onOutputLangSelected(event: ChangeEvent<HTMLSelectElement>) {
    const newOutputLang = event.currentTarget.value;
    setOutputLang(newOutputLang);
    translate(inputText, newOutputLang);
  }

  const translateWithModel = async (inputText: string, outputLang: string) => {
    const generationOptions = {
        "maxLength": 50,
        "topK": 0,
    };
    const fullInput = `translate ${inputLang} to ${outputLang}: ${inputText.trim()}`;
    const inputTokenIds = await tokenizer.current.encode(fullInput);
    async function generateProgress(outputTokenIds: number[], forInputIds: number[]) {
        let shouldContinue = true;
        return shouldContinue;
    }
    const finalOutputTokenIds = await model.current.generate(inputTokenIds, generationOptions, generateProgress);
    const finalOutput = (await tokenizer.current.decode(finalOutputTokenIds, true)).trim();
    setInputDebug(fullInput + " = " + inputTokenIds.join(" "));
    setOutputText(finalOutput);
    setStatus("Ready");
  };

  const workerRef = useRef<Worker>();
  useEffect(() => {
    workerRef.current = new Worker(new URL('../worker.ts', import.meta.url))
    workerRef.current.onmessage = (event: MessageEvent<number>) =>
      setOutputText(event.data+"");
    return () => {
      if (workerRef.current)
        workerRef.current.terminate();
    }
  }, []);
  // const handleWork = useCallback(async () => {
  //   if (workerRef.current)
  //       workerRef.current.postMessage(100000);
  // }, []);

  const translate = async (inputText: string, outputLang: string) => {
    const fullInput = `translate ${inputLang} from English to ${outputLang}: ${inputText.trim()}`;
    const command = {
      "command": "generate",
      "input": fullInput,
      "modelId": modelId,
      "modelsPath": modelsPath,
      "generationOptions": {
        "maxLength": 50,
        "topK": 0,
      }
    };
    if (workerRef.current)
        workerRef.current.postMessage(command);
  };

  return (
    <div className="content">
        <header>
            <h1>web-transformers</h1>
        </header>
        <nav>
            <a href="https://github.com/praeclarum/web-transformers">Source Code on GitHub</a>
        </nav>

        <section className="translator">
            <div className="translation-form">
                <p><b>Status</b> <span id="status">{status}</span></p>
                <div className="translation-group">
                    <div className="translation-input">
                        <p>
                            <select id="inputLang" name="inputLang" className="input" defaultValue={inputLang}>
                                <option value="English">Translate from English</option>
                            </select>
                        </p>
                        <textarea id="input" name="input" className="input"
                            onInput={onInputChanged} value={inputText}></textarea>
                        <p id="inputDebug" className="debug">{inputDebug}</p>
                    </div>
                    <div className="translation-output">
                        <p>
                            <select id="outputLang" name="outputLang" className="output" onChange={onOutputLangSelected} value={outputLang}>
                                <option value="French">Translate to French</option>
                                <option value="German">Translate to German</option>
                                <option value="Romanian">Translate to Romanian</option>
                            </select>
                        </p>
                        <p id="output" className="textarea">{outputText}</p>
                        <p id="outputDebug" className="debug"></p>
                    </div>
                </div>
                <p>This translator uses the
                    <a id="modellink" href="https://huggingface.co/t5-base">t5-base</a>
                    neural network.</p>
            </div>
        </section>
    </div>
  )
}
