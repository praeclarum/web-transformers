'use strict';

export interface TokenProcessorConfig {
  type: string;
  precompiled_charsmap: any | undefined;
  pretokenizers: TokenProcessorConfig[] | undefined;
  add_prefix_space: boolean | undefined;
  replacement: string | undefined;
  str_rep: string | undefined;
}

export class TokenProcessor {
  static fromConfig(config: TokenProcessorConfig): TokenProcessor {
    switch (config.type) {
      case 'Metaspace':
        return new MetaspaceTokenProcessor(config.add_prefix_space || false, config.replacement || '', config.str_rep);
      case 'Precompiled':
        return new PrecompiledTokenProcessor(config.precompiled_charsmap);
      case 'Sequence':
        return new SequenceTokenProcessor((config.pretokenizers || []).map((x) => TokenProcessor.fromConfig(x)));
      case 'WhitespaceSplit':
        return new WhitespaceSplitTokenProcessor();
      default:
        throw new Error('Unknown token processor type: ' + config.type);
    }
  }
  normalize(text: string): string {
    return text;
  }
  preTokenize(tokens: string[]): string[] {
    return tokens;
  }
  decodeChain(tokens: string[]): string[] {
    return tokens;
  }
}

class MetaspaceTokenProcessor extends TokenProcessor {
  addPrefixSpace: boolean;
  replacement: string;
  strRep: string;
  constructor(add_prefix_space: boolean, replacement: string, str_rep: string | undefined) {
    super();
    this.addPrefixSpace = add_prefix_space;
    this.replacement = replacement;
    this.strRep = str_rep || this.replacement;
  }
  override preTokenize(normalizedTokens: string[]) {
    const result = [];
    for (const token of normalizedTokens) {
      let normalized = token.replace(' ', this.strRep);
      if (this.addPrefixSpace && !normalized.startsWith(this.replacement)) {
        normalized = this.strRep + normalized;
      }
      result.push(normalized);
    }
    return result;
  }
  decodeChain(tokens: string[]): string[] {
    const result = [];
    let i = 0;
    for (const token of tokens) {
      let normalized = token.replace(this.replacement, ' ');
      if (this.addPrefixSpace && i == 0 && normalized.startsWith(' ')) {
        normalized = normalized.substring(1);
      }
      result.push(normalized);
      i++;
    }
    return result;
  }
}

class PrecompiledTokenProcessor extends TokenProcessor {
  charsmap: any;
  constructor(charsmap: any) {
    super();
    this.charsmap = charsmap;
  }
  override normalize(text: string) {
    return text;
  }
}

class SequenceTokenProcessor extends TokenProcessor {
  tokenizers: TokenProcessor[];
  constructor(tokenizers: TokenProcessor[]) {
    super();
    this.tokenizers = tokenizers;
  }
  override preTokenize(normalizedTokens: string[]) {
    let result = normalizedTokens;
    for (const tokenizer of this.tokenizers) {
      result = tokenizer.preTokenize(result);
    }
    return result;
  }
}

class WhitespaceSplitTokenProcessor extends TokenProcessor {
  preTokenize(normalizedTokens: string[]) {
    const result = [];
    for (const token of normalizedTokens) {
      result.push(...token.split(/\s+/));
    }
    return result;
  }
}
