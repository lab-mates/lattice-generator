Lattice Generator
=================

Makes periodic lattices (octet, bcc, gyroid and more) for resin 3D printing
and saves them as STL files. Everything runs on your own computer; nothing
is uploaded.

How to start
  1. Unzip this folder somewhere (for example, your Desktop). Do not run it
     from inside the zip.
  2. Double-click "Start Lattice Generator.bat".
     If Windows asks whether to run the file, choose Run.
  3. Your browser opens the app. Keep the black window open while you work;
     close it when you are done.

The first start takes 10 to 20 seconds while the Python engine loads.

Using it
  - Pick a shape and a lattice, set the cell size, density and number of
    cells, then press Generate.
  - Turn the model with the mouse (left drag), zoom with the wheel.
  - Press Download STL and open the file in your slicer.
  - The Printability tab warns about beams thinner than 4 printer pixels,
    trapped resin and very large files.

Tips
  - Beams is the fastest shape and makes the smallest files. Use it for big
    blocks (a 20 x 20 x 20 octet block takes about half a minute).
  - The other shapes trace a 3D grid; files grow quickly with block size.
    Keep them below a few million triangles.
  - If the black window shows an error about scripts being disabled, your
    computer blocks PowerShell scripts; ask IT, or use the web version.

Needs Windows 10 or 11 and a current Chrome, Edge or Firefox.
