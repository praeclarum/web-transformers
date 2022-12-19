import * as ort from 'onnxruntime-web';

export abstract class PretrainedModel {
  protected async loadSession(modelSource: string) {
    // console.log('Loading session from', modelSource);
    const response = await fetch(modelSource, { cache: 'force-cache' });
    const modelBuffer = await response.arrayBuffer();
    const session = await ort.InferenceSession.create(modelBuffer, { executionProviders: ['wasm'] });
    // console.log('Session loaded from', modelSource);
    return session;
  }
}
