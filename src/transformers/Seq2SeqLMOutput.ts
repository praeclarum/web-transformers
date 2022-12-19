import * as ort from 'onnxruntime-web';
import { NamedTensor } from './NamedTensor';

export class Seq2SeqLMOutput {
  readonly logits: ort.Tensor;
  readonly pastKeyValues: NamedTensor[];
  readonly encoderOutputs: ort.Tensor;
  constructor(logits: ort.Tensor, pastKeyValues: NamedTensor[], encoderOutputs: ort.Tensor) {
    this.logits = logits;
    this.pastKeyValues = pastKeyValues;
    this.encoderOutputs = encoderOutputs;
  }
}
