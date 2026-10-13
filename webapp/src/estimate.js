// Rough size estimates shown before a run (measured natively, latticegen 1.0):
// explicit struts give a fixed number of faces per cell; field meshes scale
// with N^2 per cell and grow with density.

// faces per bulk cell, 6-sided struts, by joint type
const STRUT_FACES = {
  union: { sc: 181, bcc: 336, fcc: 606, bccz: 434, fccz: 658, fbcc: 995,
           octet: 1287, diamond: 683, kelvin: 1345 },
  hull: { sc: 101, bcc: 214, fcc: 361, bccz: 250, fccz: 397, fbcc: 581,
          octet: 697, diamond: 413, kelvin: 662 },
};

// growth with the strut polygon, relative to 6 sides (round: relative to 6-sided hull)
const SIDES_FACTOR = {
  union: { 6: 1, 8: 1.2, 12: 1.6, 16: 2.1 },
  hull: { 6: 1, 8: 1.3, 12: 1.9, 16: 2.5 },
  round: { 6: 1.5, 8: 1.75, 12: 2.35, 16: 2.9 },
};

// browser: 10 M strut faces peak at ~1.4 GB of WebAssembly memory (limit 4 GB)
export const LARGE_FACES = 12_000_000;

/** latticegen.app.default_grid: half a pixel (N >= 80), a quarter pixel when instancing */
export function recommendGrid(aMin, pixel, instanced = false) {
  const N = Math.ceil(aMin / (pixel / (instanced ? 4 : 2)));
  return Math.max(80, Math.ceil(N / 8) * 8);
}

export function estimate(p) {
  const nCells = p.cells[0] * p.cells[1] * p.cells[2];
  const lengths = p.cell ? p.cell.slice(0, 3) : [p.a, p.a, p.a];
  const size = lengths.map((l, i) => l * p.cells[i]);
  let faces, N = 0;
  if (p.method === "struts") {
    const table = p.node === "union" ? STRUT_FACES.union : STRUT_FACES.hull;
    const factor = SIDES_FACTOR[p.node]?.[p.sides] ?? p.sides / 6;
    const per = (table[p.topology] ?? 800) * factor;
    faces = per * nCells;
  } else {
    N = p.N || recommendGrid(Math.min(...lengths), p.pixel, p.instanced);
    const rho = p.graded ? (p.graded[0] + p.graded[1]) / 2 : p.rho;
    let k = 4 + 50 * rho;                              // faces / (cell * N^2)
    if (p.method === "tpms" && p.sheet) k *= 1.8;
    faces = k * N * N * nCells;
  }
  if (p.plates && p.plates !== "none") {                // plates: wider footprint, taller block
    size[0] *= p.plate_size;
    size[1] *= p.plate_size;
    size[2] += p.plate_thickness * (p.plates === "both" ? 2 : 1);
  }
  return { faces, N, size, stlMB: (84 + 50 * faces) / 1e6 };
}
