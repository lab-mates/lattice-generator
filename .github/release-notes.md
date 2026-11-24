## Windows desktop app

Download **LatticeGenerator-win64-&lt;version&gt;.zip** below, unzip it anywhere and run
`LatticeGenerator\LatticeGenerator.exe`. Nothing to install: it uses WebView2,
which ships with Windows 10/11. The user manual
(`LatticeGenerator-User-Manual.pdf`) is in the same folder.

The exe is not code-signed, so the first start shows "Windows protected your PC":
click **More info → Run anyway**.

If something misbehaves, open a command prompt in that folder and run
`latticegen.exe selftest`.

## Python package

```bash
pip install "latticegen @ git+https://github.com/lab-mates/lattice-generator"
```

See the [README](https://github.com/lab-mates/lattice-generator#readme) for usage.
