# Lattice Generator offline launcher: serves this folder on http://localhost
# (this computer only) and opens it in the default browser.
# Uses only what ships with Windows; close the window to stop.
param([switch]$NoBrowser)
$ErrorActionPreference = "Stop"
$root = (Split-Path -Parent $MyInvocation.MyCommand.Path).TrimEnd("\") + "\"
$mime = @{
  ".html" = "text/html; charset=utf-8"; ".js" = "text/javascript"; ".mjs" = "text/javascript"
  ".css" = "text/css"; ".json" = "application/json"; ".wasm" = "application/wasm"
  ".zip" = "application/zip"; ".whl" = "application/zip"; ".png" = "image/png"
  ".svg" = "image/svg+xml"; ".ico" = "image/x-icon"; ".map" = "application/json"
}

$listener = $null
foreach ($port in 8765..8799) {
  $l = New-Object System.Net.HttpListener
  $l.Prefixes.Add("http://localhost:$port/")
  try { $l.Start(); $listener = $l; break } catch { $l.Close() }
}
if (-not $listener) {
  Write-Host "Could not find a free port between 8765 and 8799. Close other copies and try again."
  Read-Host "Press Enter to close"
  exit 1
}

$url = "http://localhost:$port/"
Write-Host ""
Write-Host "  Lattice Generator is running at $url"
Write-Host "  Keep this window open while you use it. Close it to stop."
Write-Host ""
if (-not $NoBrowser) { Start-Process $url }

while ($listener.IsListening) {
  $ctx = $listener.GetContext()
  $res = $ctx.Response
  try {
    $rel = [Uri]::UnescapeDataString($ctx.Request.Url.AbsolutePath).TrimStart("/")
    if ($rel -eq "") { $rel = "index.html" }
    $path = [IO.Path]::GetFullPath((Join-Path $root $rel))
    if (-not $path.StartsWith($root, [StringComparison]::OrdinalIgnoreCase) -or
        -not [IO.File]::Exists($path)) {
      $res.StatusCode = 404
    } else {
      $ext = [IO.Path]::GetExtension($path).ToLower()
      $res.ContentType = if ($mime.ContainsKey($ext)) { $mime[$ext] } else { "application/octet-stream" }
      $res.AddHeader("Cache-Control", "no-cache")
      $fs = [IO.File]::OpenRead($path)
      try { $res.ContentLength64 = $fs.Length; $fs.CopyTo($res.OutputStream) } finally { $fs.Close() }
    }
  } catch {
    # the browser closed the connection early; keep serving
  } finally {
    try { $res.Close() } catch { }
  }
}
