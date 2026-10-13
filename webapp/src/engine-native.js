// Desktop engine: native Python behind the local server of the desktop app
// (desktop/server.py).  Adds pymeshlab simplification, instancing and repair.
export class NativeEngine {
  constructor(on, token) {
    this.on = on;
    this.native = true;
    this.token = token;
    this.job = null;
  }

  async #api(path, opts = {}) {
    const res = await fetch(`api/${path}`, { ...opts, headers: { "X-Token": this.token, ...opts.headers } });
    if (!res.ok) throw new Error(`${path}: HTTP ${res.status}`);
    return res;
  }

  async start() {
    this.on.status("Starting the engine…");
    try {
      this.on.ready(await (await this.#api("info")).json());
    } catch (e) {
      this.on.error({ text: `The engine did not answer (${e.message}). Restart the app.`, detail: "" });
    }
  }

  async run(params) {
    try {
      const { job } = await (await this.#api("run", {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(params),
      })).json();
      this.job = job;
      this.#poll(job, 0);
    } catch (e) {
      this.on.error({ text: e.message, detail: "" });
    }
  }

  async #poll(job, since) {
    if (job !== this.job) return;                   // cancelled or replaced
    let s;
    try {
      s = await (await this.#api(`job/${job}?since=${since}`)).json();
    } catch (e) {
      this.on.error({ text: `Lost contact with the engine: ${e.message}`, detail: "" });
      return;
    }
    if (job !== this.job) return;
    for (const line of s.log) this.on.log(line);
    if (s.state === "running") { setTimeout(() => this.#poll(job, s.next), 300); return; }
    if (s.state === "error") { this.on.error({ text: s.error, detail: "" }); return; }
    if (s.state !== "done") return;

    const buf = await (await this.#api(`job/${job}/preview`)).arrayBuffer();
    const [nv, nf] = new Uint32Array(buf, 0, 2);
    const verts = new Float32Array(buf, 8, nv * 3);
    const faces = new Uint32Array(buf, 8 + nv * 12, nf * 3);
    const meta = s.meta;
    this.on.result({
      meta, verts, faces, sizeMB: (84 + 50 * meta.n_faces) / 1e6,
      save: () => this.#save(job, meta.filename),
    });
  }

  async #save(job, filename) {
    if (window.pywebview?.api?.save_stl) return window.pywebview.api.save_stl(job, filename);
    // --browser mode: plain download from the local server
    const a = Object.assign(document.createElement("a"),
      { href: `api/job/${job}/stl?token=${encodeURIComponent(this.token)}`, download: filename });
    document.body.append(a);
    a.click();
    a.remove();
    return { saved: filename };
  }

  async cancel() {
    const job = this.job;
    this.job = null;
    if (job) await this.#api(`job/${job}/cancel`, { method: "POST" }).catch(() => {});
    this.on.status("Cancelled.");
    this.on.idle?.();
  }
}
