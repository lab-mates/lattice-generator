// Mesh booleans for Python under Pyodide: latticegen's strut-sized joints
// (mesh_struts node="union") need a union of convex hulls, which natively comes
// from the manifold3d wheel.  Here Manifold's own WebAssembly build provides it,
// exposed to Python as the module `manifold_js` (picked up by py/bridge.py).
export async function registerManifold(py, manifoldJsUrl) {
  const { default: Module } = await import(/* @vite-ignore */ manifoldJsUrl);
  const wasm = await Module();
  wasm.setup();
  const { Manifold } = wasm;

  function hull_union(pointSets) {                 // [[[x, y, z], ...], ...]
    const hulls = pointSets.map((pts) => Manifold.hull(pts));
    const u = Manifold.union(hulls);
    const mesh = u.getMesh();
    const n = mesh.vertProperties.length / mesh.numProp;
    const verts = new Float32Array(3 * n);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < 3; k++) verts[3 * i + k] = mesh.vertProperties[i * mesh.numProp + k];
    const out = { verts, tris: mesh.triVerts.slice(), volume: u.volume() };
    hulls.forEach((h) => h.delete());
    u.delete();
    return out;
  }

  py.registerJsModule("manifold_js", { hull_union });
}
