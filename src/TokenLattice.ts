'use strict';

export class TokenLattice {
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
      for (const rnode of this.beginNodes[pos]) {
        rnode.prev = null;
        let bestScore = 0.0;
        let bestNode = null;
        for (const lnode of this.endNodes[pos]) {
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

