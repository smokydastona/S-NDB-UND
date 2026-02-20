const fs = require('fs');
const path = require('path');
const { spawnSync } = require('child_process');

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

function robocopyMirror(src, dst) {
  // robocopy exit codes: 0-7 are success (with different semantics), 8+ are failures.
  fs.mkdirSync(dst, { recursive: true });
  const args = [
    src,
    dst,
    '/MIR',
    '/R:2',
    '/W:1',
    '/NP',
    '/NFL',
    '/NDL'
  ];

  const res = spawnSync('robocopy', args, { stdio: 'inherit', windowsHide: true });
  const code = typeof res.status === 'number' ? res.status : 1;
  if (code >= 8) {
    throw new Error(`robocopy failed with exit code ${code}`);
  }
}

function copyDir(src, dst) {
  if (process.platform === 'win32') {
    // On some Windows setups (and/or newer Node versions), recursive JS copies can
    // crash the process when handling very large directory trees. Use robocopy.
    robocopyMirror(src, dst);
    return;
  }

  fs.mkdirSync(dst, { recursive: true });
  // Node 16+ supports fs.cpSync
  if (typeof fs.cpSync === 'function') {
    try {
      fs.cpSync(src, dst, { recursive: true, force: true });
      return;
    } catch (e) {
      const msg = e && e.message ? e.message : String(e);
      console.warn(`[prepare-backend] Warning: fs.cpSync failed (${msg}). Falling back to manual copy.`);
    }
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

  // Name of the PyInstaller onedir folder (and EXE base name) under dist/.
  // Note: we intentionally copy the onedir *contents* directly into electron/backend/
  // to avoid an extra nesting level that can aggravate Windows path length issues.
  const backendFolderName = process.env.SOUNDGEN_BACKEND_FOLDER || 'SÖNDBÖUND';

  // Source folder may be versioned (e.g. SÖNDBÖUND-123). Pick the newest match.
  const srcOverride = String(process.env.SOUNDGEN_BACKEND_SRC_FOLDER || '').trim();
  const srcBackend = srcOverride
    ? path.join(distDir, srcOverride)
    : (newestMatchingDir(distDir, backendFolderName) || {}).fullPath;

  const electronDir = path.join(repoRoot, 'electron');
  const dstBackendRoot = path.join(electronDir, 'backend');

  if (!srcBackend || !exists(srcBackend)) {
    console.error(`[prepare-backend] Missing backend folder under dist/.`);
    console.error(`[prepare-backend] Looked for: ${path.join(distDir, backendFolderName)} (or versioned variants)`);
    console.error(`[prepare-backend] Build it first: powershell -ExecutionPolicy Bypass -File scripts/build_exe.ps1 -Clean`);
    process.exit(2);
  }

  console.log(`[prepare-backend] Copying backend from ${srcBackend} -> ${dstBackendRoot}`);
  // Let the copy operation mirror the directory; avoid deeply-recursive deletes in JS.
  if (process.platform !== 'win32') {
    rmrf(dstBackendRoot);
  }
  copyDir(srcBackend, dstBackendRoot);

  const exe = path.join(dstBackendRoot, `${backendFolderName}.exe`);
  if (!exists(exe)) {
    console.warn(`[prepare-backend] Warning: expected EXE not found at ${exe}`);
  }

  console.log('[prepare-backend] Done');
}

try {
  main();
} catch (e) {
  console.error('[prepare-backend] Failed');
  console.error(e && e.stack ? e.stack : String(e));
  process.exit(1);
}
