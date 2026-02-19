const { app, BrowserWindow, Menu } = require('electron');
const { spawn } = require('child_process');
const fs = require('fs');
const path = require('path');

let backendProc = null;
let mainWindow = null;

function resolveBackendCommand() {
  // Preferred: use the repo venv if present.
  const repoRoot = path.resolve(__dirname, '..');
  const venvPythonWin = path.join(repoRoot, '.venv', 'Scripts', 'python.exe');
  const venvPythonPosix = path.join(repoRoot, '.venv', 'bin', 'python');

  const envPython = process.env.SOUNDGEN_PYTHON;
  if (envPython && envPython.trim()) {
    return { cmd: envPython.trim(), argsPrefix: [] };
  }

  if (process.platform === 'win32' && fs.existsSync(venvPythonWin)) {
    return { cmd: venvPythonWin, argsPrefix: [] };
  }
  if (process.platform !== 'win32' && fs.existsSync(venvPythonPosix)) {
    return { cmd: venvPythonPosix, argsPrefix: [] };
  }

  // Fallback: whatever is on PATH.
  return { cmd: 'python', argsPrefix: [] };
}

function startBackend() {
  const repoRoot = path.resolve(__dirname, '..');
  const { cmd } = resolveBackendCommand();

  const args = ['-m', 'soundgen.app', 'serve', '--host', '127.0.0.1', '--port', '0'];

  const logDir = path.join(app.getPath('userData'), 'logs');
  fs.mkdirSync(logDir, { recursive: true });
  const logPath = path.join(logDir, 'electron-backend.log');
  const logStream = fs.createWriteStream(logPath, { flags: 'a' });

  backendProc = spawn(cmd, args, {
    cwd: repoRoot,
    env: {
      ...process.env,
      // Ensures the backend doesn't try to open a browser window.
      GRADIO_ANALYTICS_ENABLED: 'False'
    },
    windowsHide: true
  });

  backendProc.stdout.setEncoding('utf8');
  backendProc.stderr.setEncoding('utf8');

  backendProc.stdout.on('data', (chunk) => {
    logStream.write(chunk);
  });
  backendProc.stderr.on('data', (chunk) => {
    logStream.write(chunk);
  });

  backendProc.on('exit', (code) => {
    logStream.write(`\n[backend exited] code=${code}\n`);
  });

  return new Promise((resolve, reject) => {
    let buffer = '';
    const timeoutMs = 30000;
    const start = Date.now();

    function tryParseUrl(text) {
      // Expected line from soundgen.serve: SOUNDGEN_URL=http://127.0.0.1:7860
      const m = text.match(/SOUNDGEN_URL=(https?:\/\/[^\s]+)/);
      if (m && m[1]) return m[1];
      return null;
    }

    const onData = (chunk) => {
      buffer += chunk;
      const url = tryParseUrl(buffer);
      if (url) {
        backendProc.stdout.off('data', onData);
        resolve(url);
      }
    };

    backendProc.stdout.on('data', onData);

    const interval = setInterval(() => {
      if (!backendProc || backendProc.killed) {
        clearInterval(interval);
        reject(new Error('Backend process terminated before URL was printed.'));
        return;
      }
      if (Date.now() - start > timeoutMs) {
        clearInterval(interval);
        reject(new Error(`Timed out waiting for backend URL. See log: ${logPath}`));
      }
    }, 250);
  });
}

function createMenu() {
  const template = [
    {
      label: 'App',
      submenu: [
        { role: 'reload' },
        { role: 'toggledevtools' },
        { type: 'separator' },
        { role: 'quit' }
      ]
    }
  ];
  const menu = Menu.buildFromTemplate(template);
  Menu.setApplicationMenu(menu);
}

async function createWindow() {
  createMenu();

  mainWindow = new BrowserWindow({
    width: 1200,
    height: 800,
    show: false,
    title: 'SÖNDBÖUND',
    webPreferences: {
      sandbox: true
    }
  });

  const url = await startBackend();
  await mainWindow.loadURL(url);
  mainWindow.show();
}

app.on('window-all-closed', () => {
  if (process.platform !== 'darwin') app.quit();
});

app.on('before-quit', () => {
  if (backendProc && !backendProc.killed) {
    try {
      backendProc.kill();
    } catch {}
  }
});

app.whenReady().then(() => {
  createWindow().catch((e) => {
    // If backend failed, show a basic error page.
    const msg = String(e && e.message ? e.message : e);
    mainWindow = new BrowserWindow({ width: 900, height: 600, title: 'SÖNDBÖUND (error)' });
    mainWindow.loadURL(
      'data:text/plain;charset=utf-8,' + encodeURIComponent('Failed to start backend.\n\n' + msg)
    );
  });
});
