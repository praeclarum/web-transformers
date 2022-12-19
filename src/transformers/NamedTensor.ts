import * as ort from 'onnxruntime-web';

export interface NamedTensor {
  name: string;
  data: ort.Tensor;
}
