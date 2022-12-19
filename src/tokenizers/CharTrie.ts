'use strict';

export class CharTrie {
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
