import * as ort from 'onnxruntime-web';

export abstract class PretrainedModel {
  static async loadSession(modelSource: string) {
    console.log('Loading session from', modelSource);
    const response = await fetch(modelSource, { cache: 'force-cache' });
    const modelBuffer = await response.arrayBuffer();
    const session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['wasm'] });
    console.log('Session loaded from', modelSource);
    return session;
  }
}

export interface GenerateOptions {
  maxLength: number;
  topK?: number;
  topP?: number;
  numBeams?: number;
}

interface NamedTensor {
  name: string;
  data: ort.Tensor;
}

export abstract class AutoModelForSeq2SeqLM extends PretrainedModel {
  private modelId: string;
  private modelsPath: string;
  private progressAsyncCallback: ((progress: number) => Promise<void>) | undefined;
  private sessions: ort.InferenceSession[] = [];

  protected constructor(
    modelId: string,
    modelsPath: string,
    progressAsyncCallback: ((progress: number) => Promise<void>) | undefined,
  ) {
    super();
    this.modelId = modelId;
    this.modelsPath = modelsPath;
    this.progressAsyncCallback = progressAsyncCallback;
  }

  static fromPretrained(
    modelId: string,
    modelsPath: string,
    progressAsyncCallback: ((progress: number) => Promise<void>) | undefined = undefined,
  ) {
    return new T5ForConditionalGeneration(modelId, modelsPath, progressAsyncCallback);
  }

  protected async getSessions() {
    if (this.sessions.length > 0) {
      return this.sessions;
    }

    const modelIdParts = this.modelId.split('/');
    const modelName = modelIdParts[modelIdParts.length - 1];
    const suffix = '-quantized';
    const encoderUrl = `${this.modelsPath}/${modelName}-encoder${suffix}.onnx`;
    const initDecoderUrl = `${this.modelsPath}/${modelName}-init-decoder${suffix}.onnx`;
    const decoderUrl = `${this.modelsPath}/${modelName}-decoder${suffix}.onnx`;

    const progressMax = 4;
    let progress = 0;
    const incrementProgress = async () => {
      progress++;
      const p = progress / progressMax;
      console.log(`Loading model ${this.modelId}... ${p * 100}%`);
      if (this.progressAsyncCallback) {
        await this.progressAsyncCallback(p);
      }
    };
    await incrementProgress();
    const encoderSessionPromise = PretrainedModel.loadSession(encoderUrl);
    const initDecoderSessionPromise = PretrainedModel.loadSession(initDecoderUrl);
    const decoderSessionPromise = PretrainedModel.loadSession(decoderUrl);
    const encoderSession = await encoderSessionPromise;
    await incrementProgress();
    const initDecoderSession = await initDecoderSessionPromise;
    await incrementProgress();
    const decoderSession = await decoderSessionPromise;
    await incrementProgress();

    this.sessions = [encoderSession, initDecoderSession, decoderSession];
    return this.sessions;
  }

  /**
   * Generate a sequence of tokens.
   *
   * @param {Array} inputTokenIds
   * @param {Object} options Properties:
   *  `maxLength` for the maximum generated sequence length,
   *  `topK` for the number of logits to consider when sampling.
   * @param {Promise} progressAsyncCallback
   * @returns The generated sequence of tokens.
   */
  async generate(
    inputTokenIds: number[],
    options: GenerateOptions,
    progressAsyncCallback:
      | ((outputTokenIds: number[], inputTokenIds: number[]) => Promise<boolean>)
      | undefined = undefined,
  ): Promise<number[]> {
    const maxLength = options.maxLength || 100;
    const topK = options.topK || 0;
    const topP = options.topP || 0;
    const numBeams = options.numBeams || 0;
    // attention_mask=token['attention_mask'], num_beams=2
    const startOfDecoderTokenId = 0;
    const endOfDecoderTokenId = 1;
    let encoderOutputs = null;
    let pastKeyValues = null;
    let outputTokenIds = [startOfDecoderTokenId];
    let numOutputTokens = 1;
    let shouldContinue = true;
    const maxOutputTokens = numOutputTokens + maxLength;
    async function progress() {
      if (progressAsyncCallback) {
        shouldContinue = await progressAsyncCallback(outputTokenIds, inputTokenIds);
      }
    }
    let sampler = (x: ort.Tensor) => this.sampleLogitsGreedily(x);
    if (topK > 0) {
      sampler = (x: ort.Tensor) => this.sampleLogitsTopK(x, topK);
    }
    while (shouldContinue && numOutputTokens < maxOutputTokens) {
      let output: Seq2SeqLMOutput = await this.forward(inputTokenIds, outputTokenIds, encoderOutputs, pastKeyValues);
      pastKeyValues = output.pastKeyValues;
      encoderOutputs = output.encoderOutputs;
      let newTokenId = sampler(output.logits);
      outputTokenIds.push(newTokenId);
      numOutputTokens++;
      await progress();
      if (newTokenId === endOfDecoderTokenId) {
        break;
      }
    }
    return outputTokenIds;
  }

  private sampleLogitsGreedily(logits: ort.Tensor): number {
    let shape = logits.dims;
    let [batchSize, seqLength, vocabSize] = shape;
    let n = batchSize * seqLength * vocabSize;
    let startIndex = n - vocabSize;
    let argmaxi = 0;
    let argmax = logits.data[startIndex + argmaxi];
    for (let i = 1; i < vocabSize; i++) {
      let l = logits.data[startIndex + i];
      if (l > argmax) {
        argmaxi = i;
        argmax = l;
      }
    }
    return argmaxi;
  }

  private sampleLogitsTopK(logits: ort.Tensor, k: number): number {
    let shape = logits.dims;
    let [batchSize, seqLength, vocabSize] = shape;
    let n = batchSize * seqLength * vocabSize;
    let startIndex = n - vocabSize;
    let logs = logits.data.slice(startIndex) as Float32Array;
    k = Math.min(k, vocabSize);
    let logitAndId = Array.from(logs)
      .map((x, i) => [x, i])
      .sort((a, b) => b[0] - a[0]);
    const sMin = Math.exp(-100.0);
    let sumS = 0.0;
    for (let i = 0; i < logitAndId.length; i++) {
      const s = i < k ? Math.exp(logitAndId[i][0]) : sMin;
      sumS += s;
      logitAndId[i][0] = s;
    }
    let r = Math.random() * sumS;
    for (let i = 0; i < logitAndId.length; i++) {
      r -= logitAndId[i][0];
      if (r <= 0) {
        return logitAndId[i][1];
      }
    }
    return logitAndId[0][1];
  }

  protected abstract forward(
    inputIds: number[],
    decoderInputIds: number[],
    encoderOutputs: ort.Tensor | null,
    pastKeyValues: NamedTensor[] | null,
  ): Promise<Seq2SeqLMOutput>;
}

class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
  constructor(
    modelId: string,
    modelsPath: string,
    progressAsyncCallback: ((progress: number) => Promise<void>) | undefined = undefined,
  ) {
    super(modelId, modelsPath, progressAsyncCallback);
  }

  protected override async forward(
    inputIds: number[],
    decoderInputIds: number[],
    encoderOutputs: ort.Tensor,
    pastKeyValues: NamedTensor[] | null,
  ) {
    const inputIdsTensor = new ort.Tensor('int64', new BigInt64Array(inputIds.map((x: number) => BigInt(x))), [
      1,
      inputIds.length,
    ]);
    const encoderAttentionMaskTensor = new ort.Tensor('int64', new BigInt64Array(inputIds.length).fill(BigInt(1)), [
      1,
      inputIds.length,
    ]);
    const [encoderSession, initDecoderSession, decoderSession] = await this.getSessions();
    if (encoderOutputs === null) {
      // console.log("Encoding...");
      const encoderFeeds = {
        input_ids: inputIdsTensor,
        attention_mask: encoderAttentionMaskTensor,
      };
      const encoderResults = await encoderSession.run(encoderFeeds);
      const encoderHiddenStates = encoderResults.hidden_states;
      encoderOutputs = encoderHiddenStates;
      // console.log("Encoding done.", encoderOutputs);
    }

    const decoderInputIdsTensor = new ort.Tensor('int64', new BigInt64Array(decoderInputIds.map((x) => BigInt(x))), [
      1,
      decoderInputIds.length,
    ]);
    // const decoderAttentionMaskTensor = new ort.Tensor("int64", new BigInt64Array(decoderInputIds.length).fill(1n), [1, decoderInputIds.length]);
    const decoderFeeds: any = {
      input_ids: decoderInputIdsTensor,
      encoder_attention_mask: encoderAttentionMaskTensor,
      encoder_hidden_states: encoderOutputs,
    };
    let logits = null;

    if (pastKeyValues === null) {
      // console.log("Init Decoding...");
      const initDecoderResults = await initDecoderSession.run(decoderFeeds);
      logits = initDecoderResults.logits;
      pastKeyValues = this.getPastKeyValues(initDecoderSession.outputNames.slice(1), initDecoderResults);
      // console.log("Init Decoding done.", logits, pastKeyValues);
    } else {
      // console.log("Decoding...");
      for (const p of pastKeyValues) {
        decoderFeeds[p.name] = p.data;
      }
      const decoderResults = await decoderSession.run(decoderFeeds);
      logits = decoderResults.logits;
      pastKeyValues = this.getPastKeyValues(decoderSession.outputNames.slice(1), decoderResults);
      // console.log("Decoding done.", logits, pastKeyValues);
    }
    return new Seq2SeqLMOutput(logits, pastKeyValues, encoderOutputs);
  }

  private getPastKeyValues(pkvNames: string[], decoderResults: ort.InferenceSession.OnnxValueMapType): NamedTensor[] {
    const pkvs: NamedTensor[] = [];
    for (const i in pkvNames) {
      const k = pkvNames[i];
      const v = decoderResults[k] as ort.Tensor;
      pkvs.push({ name: `pkv_${i}`, data: v });
    }
    return pkvs;
  }
}

class Seq2SeqLMOutput {
  readonly logits: ort.Tensor;
  readonly pastKeyValues: NamedTensor[];
  readonly encoderOutputs: ort.Tensor;
  constructor(logits: ort.Tensor, pastKeyValues: NamedTensor[], encoderOutputs: ort.Tensor) {
    this.logits = logits;
    this.pastKeyValues = pastKeyValues;
    this.encoderOutputs = encoderOutputs;
  }
}
