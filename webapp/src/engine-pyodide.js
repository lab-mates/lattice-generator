// In-browser engine: latticegen under Pyodide in a Web Worker (web version).
export class PyodideEngine {
  constructor(on) {
    this.on = on;               // {status, log, ready, result, error}
    this.native = false;
    this.worker = null;
  }

  start() {
    this.worker = new Worker(new URL("./worker.js", import.meta.url), { type: "module" });
    this.worker.onmessage = ({ data }) => this.#message(data);
    this.worker.onerror = (e) => this.on.error({ text: `The Python engine stopped: ${e.message || "unknown error"}`,
                                                 detail: "", fatal: true });
    this.worker.postMessage({ type: "init", base: new URL("./", document.baseURI).href });
    this.on.status("Starting the Python engine…");
  }

  #message(data) {
    if (data.type === "status") this.on.status(data.text);
    else if (data.type === "log") this.on.log(data.text);
    else if (data.type === "ready") this.on.ready(data.options);
    else if (data.type === "error") this.on.error({ ...data, fatal: /MemoryError|Aborted/i.test(data.text) });
    else if (data.type === "result") {
      const blob = new Blob([data.stl], { type: "model/stl" });
      const filename = data.meta.filename;
      this.on.result({
        meta: data.meta, verts: data.verts, faces: data.faces, sizeMB: blob.size / 1e6,
        save: async () => { saveBlob(blob, filename); return { saved: filename }; },
      });
    }
  }

  run(params) { this.worker.postMessage({ type: "run", params }); }

  /** Stop a run (or recover from a crash): the interpreter is restarted. */
  cancel() {
    this.worker.terminate();
    this.start();
    this.on.status("Restarting the Python engine…");
  }
}

export function saveBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = Object.assign(document.createElement("a"), { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 10_000);
}
