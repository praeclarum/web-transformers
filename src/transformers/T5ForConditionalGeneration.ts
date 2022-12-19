import * as ort from 'onnxruntime-web';
import { AutoModelForSeq2SeqLM } from './AutoModelForSeq2SeqLM';
import { NamedTensor } from './NamedTensor';
import { Seq2SeqLMOutput } from './Seq2SeqLMOutput';

export class T5ForConditionalGeneration extends AutoModelForSeq2SeqLM {
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
