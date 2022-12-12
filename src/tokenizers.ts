'use strict';

interface TokenizerConfig {
  model: {
    vocab: (string | number)[][];
    unk_id: number;
  };
  pre_tokenizer: TokenProcessorConfig;
  normalizer: TokenProcessorConfig;
  decoder: TokenProcessorConfig;
  added_tokens: any[];
}

interface TokenProcessorConfig {
  type: string;
  precompiled_charsmap: any | undefined;
  pretokenizers: TokenProcessorConfig[] | undefined;
  add_prefix_space: boolean | undefined;
  replacement: string | undefined;
  str_rep: string | undefined;
}

export class AutoTokenizer {
  private tokenizer: Tokenizer|null = null;
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

export async function loadTokenizer(url: string): Promise<Tokenizer> {
  const response = await fetch(url);
  const jtokenizer = await response.json();
  return Tokenizer.fromConfig(jtokenizer);
}

export class Tokenizer {
  bosToken: string;
  bosTokenId: number;
  eosToken: string;
  eosTokenId: number;
  unkToken: string;
  unkTokenId: number;
  private vocab: (string | number)[][];
  private specialTokens: any[];
  private specialTokenIds: Map<number, number>;
  private trie: CharTrie;
  private tokenToIds: Map<string, number>;
  private minScore: number;
  private unkScore: number;
  private preTokenizer: TokenProcessor;
  private normalizer: TokenProcessor;
  private decoder: TokenProcessor;
  constructor(
    vocab: (string | number)[][],
    unkTokenId: number,
    specialTokens: any[],
    normalizer: TokenProcessor,
    preTokenizer: TokenProcessor,
    decoder: TokenProcessor,
  ) {
    this.vocab = vocab;
    this.unkTokenId = unkTokenId;
    this.specialTokens = specialTokens;
    this.specialTokenIds = new Map(specialTokens.map((x) => [x.id, x]));
    this.normalizer = normalizer;
    this.preTokenizer = preTokenizer;
    this.decoder = decoder;
    this.tokenToIds = new Map(vocab.map((x, i) => [this.normalize(String(x[0])), i]));
    this.bosToken = this.normalize(' ');
    this.bosTokenId = this.getTokenId(this.bosToken);
    this.eosToken = '</s>';
    this.eosTokenId = this.getTokenId(this.eosToken);
    this.unkToken = String(this.vocab[this.unkTokenId][0]);
    this.trie = new CharTrie();
    this.minScore = 1.0e6;
    vocab.forEach((x) => (this.minScore = Math.min(this.minScore, Number(x[1]))));
    this.unkScore = this.minScore - 10.0;
    vocab[unkTokenId][1] = this.unkScore;
    vocab.forEach((x) => this.trie.push(String(x[0])));
  }
  static fromConfig(config: TokenizerConfig) {
    const preTokenizer: TokenProcessor = TokenProcessor.fromConfig(config.pre_tokenizer);
    const normalizer: TokenProcessor = TokenProcessor.fromConfig(config.normalizer);
    const decoder: TokenProcessor = TokenProcessor.fromConfig(config.decoder);
    return new Tokenizer(
      config.model.vocab,
      config.model.unk_id,
      config.added_tokens,
      normalizer,
      preTokenizer,
      decoder,
    );
  }
  getTokenId(normalizedToken: string): number {
    return this.tokenToIds.get(normalizedToken) || this.unkTokenId;
  }
  private normalize(text: string): string {
    return this.normalizer.normalize(text);
  }
  private preTokenize(normalized: string[]) {
    return this.preTokenizer.preTokenize(normalized);
  }
  private populateNodes(lattice: TokenLattice) {
    const sentence = lattice.sentence;
    const len = sentence.length;
    let beginPos = 0;
    while (beginPos < len) {
      const mblen = 1;
      let hasSingleNode = false;
      const tokens = [];
      for (let token of this.trie.commonPrefixSearch(sentence.slice(beginPos))) {
        tokens.push(token);
        const tokenId = this.getTokenId(token);
        const tokenScore = Number(this.vocab[tokenId][1]);
        const n = token.length;
        lattice.insert(beginPos, n, tokenScore, tokenId);
        if (!hasSingleNode && n == mblen) {
          hasSingleNode = true;
        }
      }
      if (!hasSingleNode) {
        lattice.insert(beginPos, mblen, this.unkScore, this.unkTokenId);
      }
      beginPos += mblen;
    }
  }
  private tokenize(normalized: string) {
    const lattice = new TokenLattice(normalized, this.bosTokenId, this.eosTokenId);
    this.populateNodes(lattice);
    const tokenIds = lattice.tokenIds();
    return tokenIds;
  }
  encode(text: string): number[] {
    if (text === null || text === undefined || text.length === 0) return [this.eosTokenId];
    const normalized = this.normalize(text);
    const pre = this.preTokenize([normalized]);
    const tokens = [];
    for (let token of pre) {
      const tokenized = this.tokenize(token);
      tokens.push(...tokenized);
    }
    tokens.push(this.eosTokenId);
    return tokens;
  }
  decode(tokenIds: number[], skipSpecialTokens: boolean): string {
    const tokens = tokenIds.map((x) => {
      if (this.specialTokenIds.get(x) !== undefined && skipSpecialTokens) {
        return '';
      } else if (x == this.unkTokenId) {
        return this.unkToken + ' ';
      } else if (x in this.vocab) {
        return String(this.vocab[x][0]);
      } else {
        return `[${x}]`;
      }
    });
    const decodedTokens = this.decoder.decodeChain(tokens);
    const decoded = decodedTokens.join('');
    return decoded;
  }
}

class CharTrieNode {
  isLeaf: boolean;
  children: Map<string, CharTrieNode>;
  constructor(isLeaf: boolean, children: Map<string, CharTrieNode>) {
    this.isLeaf = isLeaf;
    this.children = children;
  }
  static default(): CharTrieNode {
    return new CharTrieNode(false, new Map());
  }
}

class CharTrie {
  private readonly root: CharTrieNode;
  constructor() {
    this.root = CharTrieNode.default();
  }
  push(text: string) {
    let node = this.root;
    for (let i = 0; i < text.length; i++) {
      const ch = text[i];
      let child = node.children.get(ch);
      if (child === undefined) {
        child = CharTrieNode.default();
        node.children.set(ch, child);
      }
      node = child;
    }
    node.isLeaf = true;
  }
  *commonPrefixSearch(text: string) {
    let node: CharTrieNode | undefined = this.root;
    let prefix = '';
    for (let i = 0; i < text.length && node !== undefined; i++) {
      const ch = text[i];
      prefix += ch;
      node = node.children.get(ch);
      if (node !== undefined && node.isLeaf) {
        yield prefix;
      }
    }
  }
}

class TokenLattice {
  readonly sentence: string;
  private nodes: TokenLatticeNode[];
  private len: number;
  private bosTokenId: number;
  private eosTokenId: number;
  private beginNodes: TokenLatticeNode[][];
  private endNodes: TokenLatticeNode[][];

  constructor(sentence: string, bosTokenId: number, eosTokenId: number) {
    this.sentence = sentence;
    this.len = sentence.length;
    this.bosTokenId = bosTokenId;
    this.eosTokenId = eosTokenId;
    this.nodes = [];
    this.beginNodes = new Array(this.len + 1);
    this.endNodes = new Array(this.len + 1);
    for (let i = 0; i < this.len + 1; i++) {
      this.beginNodes[i] = [];
      this.endNodes[i] = [];
    }
    const bos = new TokenLatticeNode(this.bosTokenId, 0, 0, 0, 0.0);
    const eos = new TokenLatticeNode(this.eosTokenId, 1, this.len, 0, 0.0);
    this.nodes.push(bos.clone());
    this.nodes.push(eos.clone());
    this.beginNodes[this.len].push(eos);
    this.endNodes[0].push(bos);
  }
  insert(pos: number, length: number, score: number, tokenId: number) {
    const nodeId = this.nodes.length;
    const node = new TokenLatticeNode(tokenId, nodeId, pos, length, score);
    this.beginNodes[pos].push(node);
    this.endNodes[pos + length].push(node);
    this.nodes.push(node);
  }
  viterbi(): TokenLatticeNode[] {
    const len = this.len;
    let pos = 0;
    while (pos <= len) {
      if (this.beginNodes[pos].length == 0) {
        return [];
      }
      for (let rnode of this.beginNodes[pos]) {
        rnode.prev = null;
        let bestScore = 0.0;
        let bestNode = null;
        for (let lnode of this.endNodes[pos]) {
          const score = lnode.backtraceScore + rnode.score;
          if (bestNode === null || score > bestScore) {
            bestNode = lnode.clone();
            bestScore = score;
          }
        }
        if (bestNode !== null) {
          rnode.prev = bestNode;
          rnode.backtraceScore = bestScore;
        } else {
          return [];
        }
      }
      pos++;
    }
    const results = [];
    const root = this.beginNodes[len][0];
    const prev = root.prev;
    if (prev === null) {
      return [];
    }
    let node: TokenLatticeNode | null = prev.clone();
    while (node !== null && node.prev !== null) {
      results.push(node.clone());
      const n = node.clone();
      node = n.prev !== null ? n.prev.clone() : null;
    }
    results.reverse();
    return results;
  }
  piece(node: TokenLatticeNode) {
    return this.sentence.slice(node.pos, node.pos + node.length);
  }
  tokens() {
    const nodes = this.viterbi();
    return nodes.map((x) => this.piece(x));
  }
  tokenIds() {
    const nodes = this.viterbi();
    return nodes.map((x) => x.tokenId);
  }
}
class TokenLatticeNode {
  readonly tokenId: number;
  readonly nodeId: number;
  readonly pos: number;
  readonly length: number;
  readonly score: number;
  prev: TokenLatticeNode | null;
  backtraceScore: number;
  constructor(tokenId: number, nodeId: number, pos: number, length: number, score: number) {
    this.tokenId = tokenId;
    this.nodeId = nodeId;
    this.pos = pos;
    this.length = length;
    this.score = score;
    this.prev = null;
    this.backtraceScore = 0.0;
  }
  clone() {
    const n = new TokenLatticeNode(this.tokenId, this.nodeId, this.pos, this.length, this.score);
    n.prev = this.prev;
    n.backtraceScore = this.backtraceScore;
    return n;
  }
}

class TokenProcessor {
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
    for (let token of normalizedTokens) {
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
    for (let token of tokens) {
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
    for (let tokenizer of this.tokenizers) {
      result = tokenizer.preTokenize(result);
    }
    return result;
  }
}

class WhitespaceSplitTokenProcessor extends TokenProcessor {
  preTokenize(normalizedTokens: string[]) {
    const result = [];
    for (let token of normalizedTokens) {
      result.push(...token.split(/\s+/));
    }
    return result;
  }
}
