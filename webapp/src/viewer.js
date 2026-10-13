// Three.js build-plate view: the lattice in amber resin on a millimetre grid.
// latticegen works in mm with z as the build direction; three.js is y-up, so
// the model group is rotated -90 deg about x.
import * as THREE from "three";
import { OrbitControls } from "three/addons/controls/OrbitControls.js";

export class Viewer {
  constructor(container) {
    this.container = container;
    this.renderer = new THREE.WebGLRenderer({ antialias: true, alpha: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    container.appendChild(this.renderer.domElement);

    this.scene = new THREE.Scene();
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.01, 10000);
    this.controls = new OrbitControls(this.camera, this.renderer.domElement);
    this.controls.enableDamping = true;

    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a8f99, 1.6));
    this.key = new THREE.DirectionalLight(0xffffff, 1.8);
    this.scene.add(this.key);
    this.camera.add(this.key);
    this.key.position.set(1, 2, 1.5);
    this.scene.add(this.camera);

    this.model = new THREE.Group();
    this.model.rotation.x = -Math.PI / 2;
    this.scene.add(this.model);
    // polygon offset pushes the faces back so the edge lines draw on top
    this.material = new THREE.MeshStandardMaterial({
      color: 0xe3a008, roughness: 0.42, metalness: 0.0, flatShading: true,
      polygonOffset: true, polygonOffsetFactor: 1, polygonOffsetUnits: 1,
    });
    this.edgeMaterial = new THREE.MeshBasicMaterial({
      wireframe: true, transparent: true, opacity: 0.35, depthWrite: false,
    });
    this.grid = null;
    this.mesh = null;
    this.edges = null;
    this.setColor(0xe3a008);

    new ResizeObserver(() => this.resize()).observe(container);
    this.resize();
    const loop = () => {
      this.controls.update();
      this.renderer.render(this.scene, this.camera);
      requestAnimationFrame(loop);
    };
    loop();
  }

  resize() {
    const { clientWidth: w, clientHeight: h } = this.container;
    if (!w || !h) return;
    this.renderer.setSize(w, h, false);
    this.camera.aspect = w / h;
    this.camera.updateProjectionMatrix();
  }

  setGridColors(major, minor) {
    this.gridColors = [major, minor];
    if (this.grid) this.buildGrid(this.gridSize);
  }

  buildGrid(size) {
    if (this.grid) { this.scene.remove(this.grid); this.grid.geometry.dispose(); }
    this.gridSize = size;
    // 1 mm lines up to 60 mm plates, else 5 mm; always an even count
    const step = size <= 60 ? 1 : 5;
    const n = Math.max(2, Math.ceil(size / step / 2) * 2);
    const [major, minor] = this.gridColors || [0x8b95a1, 0xc3c9d0];
    this.grid = new THREE.GridHelper(n * step, n, major, minor);
    this.grid.position.y = -0.001;
    this.scene.add(this.grid);
  }

  /** lattice colour; the edge lines take a contrasting shade of it */
  setColor(hex) {
    this.material.color.setHex(hex);
    const dark = this.material.color.getHSL({}).l < 0.3;
    this.edgeMaterial.color.setHex(dark ? 0xffffff : 0x000000);
    this.edgeMaterial.opacity = dark ? 0.3 : 0.35;
  }

  /** smooth (Gouraud-style, interpolated vertex normals) or flat shading */
  setSmooth(on) {
    this.smooth = on;
    this.material.flatShading = !on;
    this.material.needsUpdate = true;
    if (on && this.mesh && !this.mesh.geometry.attributes.normal)
      this.mesh.geometry.computeVertexNormals();
  }

  /** triangle edges drawn over the shaded faces */
  setEdges(on) {
    this.showEdges = on;
    if (this.edges) this.edges.visible = on;
    else if (on && this.mesh) this.addEdges();
  }

  addEdges() {
    this.edges = new THREE.Mesh(this.mesh.geometry, this.edgeMaterial);
    this.edges.renderOrder = 1;
    this.model.add(this.edges);
  }

  show(verts, faces) {
    this.clear();
    const g = new THREE.BufferGeometry();
    g.setAttribute("position", new THREE.BufferAttribute(verts, 3));
    g.setIndex(new THREE.BufferAttribute(faces, 1));
    g.computeBoundingBox();
    const bb = g.boundingBox;
    // sit the block on the plate, centred on the origin
    const c = bb.getCenter(new THREE.Vector3());
    g.translate(-c.x, -c.y, -bb.min.z);
    g.computeBoundingSphere();
    if (this.smooth) g.computeVertexNormals();
    this.mesh = new THREE.Mesh(g, this.material);
    this.model.add(this.mesh);
    if (this.showEdges) this.addEdges();

    const size = bb.getSize(new THREE.Vector3());
    this.buildGrid(Math.max(size.x, size.y) * 1.6);
    this.frame(size);
  }

  frame(size) {
    const r = Math.max(size.length() / 2, 1e-3);
    const dist = r / Math.sin((this.camera.fov * Math.PI) / 360) * 1.1;
    const target = new THREE.Vector3(0, size.z / 2, 0);
    const dir = new THREE.Vector3(1, 0.75, 1.25).normalize();
    this.camera.position.copy(target).addScaledVector(dir, dist);
    this.camera.near = dist / 100;
    this.camera.far = dist * 100;
    this.camera.updateProjectionMatrix();
    this.controls.target.copy(target);
    this.controls.update();
  }

  /** empty plate framed for a block of ``size`` mm */
  reset(size = 8) {
    this.clear();
    this.buildGrid(size * 2);
    this.frame(new THREE.Vector3(size, size, size));
  }

  clear() {
    if (this.edges) { this.model.remove(this.edges); this.edges = null; }
    if (this.mesh) { this.model.remove(this.mesh); this.mesh.geometry.dispose(); this.mesh = null; }
  }
}
