import { Tokenizer } from '../index';

import * as t5_config from './t5-base-tokenizer.json';

describe('t5 tokenizer', () => {
  let tests: (string | number[])[][];

  beforeAll(() => {
    tests = require('./t5-base-tests.json');
  });

  test('encode and decode', () => {
    const maxTests = 100;
    expect(tests.length).toBeGreaterThanOrEqual(maxTests);

    let numEncode = 0;
    let numEncodePass = 0;
    let numDecode = 0;
    let numDecodePass = 0;

    for (let i = 0; i <= maxTests; i++) {
      const test = tests[i];
      const inputText = String(test[0]);
      const expectedEncode: number[] = test[1] as any;
      const expectedDecode = String(test[2]);
      const tokenizer = Tokenizer.fromConfig(t5_config);
      let testEncode: number[] = [];
      let encError = '';
      let decError = '';
      try {
        testEncode = tokenizer.encode(inputText);
      } catch (e: any) {
        encError = String(e.message);
      }
      let testDecode = '';
      try {
        testDecode = tokenizer.decode(testEncode, false);
      } catch (e: any) {
        decError = String(e.message);
      }
      const decodePass = testDecode === expectedDecode;
      numDecode += 1;
      numDecodePass += decodePass ? 1 : 0;
      let encodePass = testEncode.length === expectedEncode.length;
      if (encodePass) {
        for (let i = 0; i < testEncode.length; i++) {
          if (testEncode[i] !== expectedEncode[i]) {
            encodePass = false;
            break;
          }
        }
      }
      numEncode += 1;
      numEncodePass += encodePass ? 1 : 0;
      const pass = decodePass && encodePass;
    }

    const encPassRatio = numEncodePass / numEncode;
    expect(encPassRatio).toBeGreaterThan(0.92);
    const decPassRatio = numDecodePass / numDecode;
    expect(decPassRatio).toBeGreaterThan(0.85);
  });
});
