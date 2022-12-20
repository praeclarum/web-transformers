import { useState, useRef, FormEvent, ChangeEvent, useEffect } from 'react';

export default function Home() {

  const modelId = "t5-small";
  const modelsPath = "/models";

  const [inputLang, setInputLang] = useState('English');
  const [outputLang, setOutputLang] = useState('French');
  const [inputText, setInputText] = useState('The universe is a dark forest.');
  const [outputText, setOutputText] = useState('');

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

  const workerRef = useRef<Worker>();
  useEffect(() => {
    workerRef.current = new Worker(new URL('../worker.ts', import.meta.url));
    workerRef.current.onmessage = (event: MessageEvent<any>) => {
      setOutputText(event.data.outputText);
    };
    translate(inputText, outputLang);
    return () => {
      if (workerRef.current)
        workerRef.current.terminate();
    }
  }, []);

  const translate = async (inputText: string, outputLang: string) => {
    const fullInput = `translate ${inputLang} to ${outputLang}: ${inputText.trim()}`;
    const command = {
      "inputText": fullInput,
      "modelId": modelId,
      "modelsPath": modelsPath,
      "maxLength": 50,
      "topK": 0,
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
                <div className="translation-group">
                    <div className="translation-input">
                        <p>
                            <select id="inputLang" name="inputLang" className="input" defaultValue={inputLang}>
                                <option value="English">Translate from English</option>
                            </select>
                        </p>
                        <textarea id="input" name="input" className="input"
                            onInput={onInputChanged} value={inputText}></textarea>
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
