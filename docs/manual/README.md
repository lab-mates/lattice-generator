# Desktop app user manual

`manual.html` is the source of `LatticeGenerator-User-Manual.pdf`, written for
A4 print. To rebuild the PDF, open `manual.html` in Chrome or Edge and print it
to PDF with "Background graphics" on and margins set to "Default", or print it
headlessly through the DevTools protocol (`Page.printToPDF` with
`preferCSSPageSize` and `printBackground`).

The screenshots in `img/` were taken from the desktop app's own page and
engine (`python -m desktop --browser`) in a 1400 x 900 window at 1.5x scale,
with the default settings of version 1.0. Retake them if the interface
changes.
