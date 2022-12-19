'use strict';
import { Tokenizer } from './Tokenizer';

export class AutoTokenizer {
  private tokenizer: Tokenizer | null = null;
  private modelId: string;
  private modelsPath: string;

  constructor(modelId: string, modelsPath: string) {
    this.modelId = modelId;
    this.modelsPath = modelsPath;
  }

  static fromPretrained(modelId: string, modelsPath: string): AutoTokenizer {
    return new AutoTokenizer(modelId, modelsPath);
  }

  private async load(): Promise<Tokenizer> {
    if (this.tokenizer != null) {
      return this.tokenizer;
    }
    const modelIdParts = this.modelId.split('/');
    const modelName = modelIdParts[modelIdParts.length - 1];
    const url = `${this.modelsPath}/${modelName}-tokenizer.json`;
    const response = await fetch(url);
    this.tokenizer = Tokenizer.fromConfig(await response.json());
    return this.tokenizer;
  }

  async encode(text: string): Promise<number[]> {
    const tokenizer = await this.load();
    return tokenizer.encode(text);
  }

  async decode(tokenIds: number[], skipSpecialTokens: boolean): Promise<string> {
    const tokenizer = await this.load();
    return tokenizer.decode(tokenIds, skipSpecialTokens);
  }
}
