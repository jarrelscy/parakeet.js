// Simple text tokenizer/decoder for Parakeet models (browser-friendly, fetch-only).

/**
 * Fetch a text file (tokens.txt or vocab.txt) and return its contents.
 * @param {string} url Remote URL or relative path served by the web app.
 */
async function fetchText(url) {
  const resp = await fetch(url);
  if (!resp.ok) throw new Error(`Failed to fetch ${url}: ${resp.status}`);
  return resp.text();
}

export class ParakeetTokenizer {
  /**
   * @param {string[]} id2token Array where index=id and value=token string
   */
  constructor(id2token) {
    this.id2token = id2token;
    this.blankToken = '<blk>';
  }

  static async fromUrl(tokensUrl) {
    const text = await fetchText(tokensUrl);
    const trimmed = text.trim();
    let id2token = [];

    if (trimmed.startsWith('{')) {
      const json = JSON.parse(trimmed);
      const vocab = json?.model?.vocab;
      if (Array.isArray(vocab)) {
        id2token = vocab.map((entry) => entry[0]);
      } else if (vocab && typeof vocab === 'object') {
        for (const [token, id] of Object.entries(vocab)) {
          id2token[id] = token;
        }
      }
    }

    if (!id2token.length) {
      const lines = text.split(/\r?\n/).filter(Boolean);
      for (const line of lines) {
        const [tok, idStr] = line.split(/\s+/);
        const id = parseInt(idStr, 10);
        id2token[id] = tok;
      }
    }
    const tokenizer = new ParakeetTokenizer(id2token);
    const blankCandidates = ['<epsilon>', '<blank>', '<blk>'];
    for (const cand of blankCandidates) {
      if (id2token.includes(cand)) {
        tokenizer.blankToken = cand;
        break;
      }
    }
    return tokenizer;
  }

  /**
   * Decode an array of token IDs into a human readable string.
   * Implements the SentencePiece rule where leading `▁` marks a space.
   * @param {number[]} ids
   * @returns {string}
   */
  decode(ids, options = {}) {
    const skipTokens = new Set(options.skipTokens || []);
    skipTokens.add(this.blankToken);
    const pieces = [];
    for (const id of ids) {
      const token = this.id2token[id];
      if (token === undefined || skipTokens.has(token)) continue;
      pieces.push(token.replace(/\u2581/g, ' '));
    }

    const raw = pieces.join('');
    if (!raw) return '';

    // Mirror the spacing cleanup implemented in onnx_asr so our outputs
    // match the Python reference decoder byte-for-byte.
    return raw.replace(/(^\s|\s\B|(\s)\b)/g, (_, _discard, keep) => (keep ? ' ' : ''));
  }
}
