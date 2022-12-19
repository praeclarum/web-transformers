import { AutoModelForSeq2SeqLM } from '../index';

describe('transformers', () => {
  test('nop', () => {
    expect(AutoModelForSeq2SeqLM).toBeTruthy();
  });
});
