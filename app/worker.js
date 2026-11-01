// Inference worker: owns the ONNX sessions so detection never blocks taps, scrolling or the overlay.
import { createPipeline } from './pipeline.js';

const pipe = createPipeline();
self.onmessage = async ({ data: { id, op, arg } }) => {
  try {
    const res = op === 'load' ? await pipe.load(s => self.postMessage({ status: s })) : await pipe[op](arg);
    self.postMessage({ id, res });
  } catch (e) {
    self.postMessage({ id, err: String(e?.message || e) });
  }
};
