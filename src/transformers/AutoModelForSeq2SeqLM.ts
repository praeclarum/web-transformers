import * as ort from 'onnxruntime-web';
import { GenerateOptions } from './GenerateOptions';
import { PretrainedModel } from './PretrainedModel';
import { NamedTensor } from './NamedTensor';
import { Seq2SeqLMOutput } from './Seq2SeqLMOutput';

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
    const outputTokenIds = [startOfDecoderTokenId];
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
      const output: Seq2SeqLMOutput = await this.forward(inputTokenIds, outputTokenIds, encoderOutputs, pastKeyValues);
      pastKeyValues = output.pastKeyValues;
      encoderOutputs = output.encoderOutputs;
      const newTokenId = sampler(output.logits);
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
    const shape = logits.dims;
    const [batchSize, seqLength, vocabSize] = shape;
    const n = batchSize * seqLength * vocabSize;
    const startIndex = n - vocabSize;
    let argmaxi = 0;
    let argmax = logits.data[startIndex + argmaxi];
    for (let i = 1; i < vocabSize; i++) {
      const l = logits.data[startIndex + i];
      if (l > argmax) {
        argmaxi = i;
        argmax = l;
      }
    }
    return argmaxi;
  }

  private sampleLogitsTopK(logits: ort.Tensor, k: number): number {
    const shape = logits.dims;
    const [batchSize, seqLength, vocabSize] = shape;
    const n = batchSize * seqLength * vocabSize;
    const startIndex = n - vocabSize;
    const logs = logits.data.slice(startIndex) as Float32Array;
    k = Math.min(k, vocabSize);
    const logitAndId = Array.from(logs)
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
