const fs = require('fs');
const path = require('path');

function exists(p) {
  try {
    fs.accessSync(p);
    return true;
  } catch {
    return false;
  }
}

function rmrf(p) {
  if (!exists(p)) return;
  fs.rmSync(p, { recursive: true, force: true });
}

function copyDir(src, dst) {
  fs.mkdirSync(dst, { recursive: true });
  // Node 16+ supports fs.cpSync
  if (typeof fs.cpSync === 'function') {
    fs.cpSync(src, dst, { recursive: true });
    return;
  }
  // Fallback: manual walk
  for (const entry of fs.readdirSync(src, { withFileTypes: true })) {
    const s = path.join(src, entry.name);
    const d = path.join(dst, entry.name);
    if (entry.isDirectory()) {
      copyDir(s, d);
    } else if (entry.isFile()) {
      fs.copyFileSync(s, d);
    }
  }
}

function listDirs(dir) {
  try {
    return fs.readdirSync(dir, { withFileTypes: true })
      .filter((e) => e.isDirectory())
      .map((e) => e.name);
  } catch {
    return [];
  }
}

function newestMatchingDir(parentDir, prefix) {
  const dirs = listDirs(parentDir)
    .filter((n) => n === prefix || n.startsWith(prefix + '-'))
    .map((n) => {
      const p = path.join(parentDir, n);
      let m = 0;
      try { m = fs.statSync(p).mtimeMs || 0; } catch {}
      return { name: n, fullPath: p, mtimeMs: m };
    })
    .sort((a, b) => (b.mtimeMs - a.mtimeMs));

  return dirs.length > 0 ? dirs[0] : null;
}

function main() {
  const repoRoot = path.resolve(__dirname, '..', '..');
  const distDir = path.join(repoRoot, 'dist');

  // Destination folder name (stable) used by Electron runtime.
  const backendFolderName = process.env.SOUNDGEN_BACKEND_FOLDER || 'SÖNDBÖUND';

  // Source folder may be versioned (e.g. SÖNDBÖUND-123). Pick the newest match.
  const srcOverride = String(process.env.SOUNDGEN_BACKEND_SRC_FOLDER || '').trim();
  const srcBackend = srcOverride
    ? path.join(distDir, srcOverride)
    : (newestMatchingDir(distDir, backendFolderName) || {}).fullPath;

  const electronDir = path.join(repoRoot, 'electron');
  const dstBackendRoot = path.join(electronDir, 'backend');
  const dstBackend = path.join(dstBackendRoot, backendFolderName);

  if (!srcBackend || !exists(srcBackend)) {
    console.error(`[prepare-backend] Missing backend folder under dist/.`);
    console.error(`[prepare-backend] Looked for: ${path.join(distDir, backendFolderName)} (or versioned variants)`);
    console.error(`[prepare-backend] Build it first: powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1 -Clean`);
    process.exit(2);
  }

  console.log(`[prepare-backend] Copying backend from ${srcBackend} -> ${dstBackend}`);
  rmrf(dstBackend);
  copyDir(srcBackend, dstBackend);

  const exe = path.join(dstBackend, `${backendFolderName}.exe`);
  if (!exists(exe)) {
    console.warn(`[prepare-backend] Warning: expected EXE not found at ${exe}`);
  }

  console.log('[prepare-backend] Done');
}

main();
