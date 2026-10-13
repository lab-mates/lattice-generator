// Mesh booleans for Python under Pyodide: latticegen's strut-sized joints
// (mesh_struts node="union") need a union of convex hulls, and add_plates a
// union of the block with its plates; natively both come from the manifold3d wheel.  Here Manifold's own WebAssembly build provides it,
// exposed to Python as the module `manifold_js` (picked up by py/bridge.py).
export async function registerManifold(py, manifoldJsUrl) {
  const { default: Module } = await import(/* @vite-ignore */ manifoldJsUrl);
  const wasm = await Module();
  wasm.setup();
  const { Manifold, Mesh } = wasm;

  function toArrays(u) {
    const mesh = u.getMesh();
    const n = mesh.vertProperties.length / mesh.numProp;
    const verts = new Float32Array(3 * n);
    for (let i = 0; i < n; i++)
      for (let k = 0; k < 3; k++) verts[3 * i + k] = mesh.vertProperties[i * mesh.numProp + k];
    return { verts, tris: mesh.triVerts.slice(), faceID: mesh.faceID.slice() };
  }

  function hull_union(pointSets) {                 // [[[x, y, z], ...], ...]
    const hulls = pointSets.map((pts) => Manifold.hull(pts));
    const u = Manifold.union(hulls);
    const out = { ...toArrays(u), volume: u.volume() };
    hulls.forEach((h) => h.delete());
    u.delete();
    return out;
  }

  // verts: Float32Array (3n), tris: Uint32Array (3m), faceID: Uint32Array (m),
  // plates: [[[x, y, z] x 8], ...]; faceID comes back on the faces derived from them
  function union_plates(verts, tris, faceID, plates) {
    const parts = [];
    try {
      parts.push(new Manifold(new Mesh({ numProp: 3, vertProperties: verts, triVerts: tris, faceID })));
    } catch (e) {
      return { error: `the mesh is not a closed solid (${e.message || e}); repair it before adding plates` };
    }
    for (const p of plates) parts.push(Manifold.hull(p));
    const u = Manifold.union(parts);
    const out = toArrays(u);
    parts.forEach((m) => m.delete());
    u.delete();
    return { ...out, error: null };
  }

  py.registerJsModule("manifold_js", { hull_union, union_plates });
}
