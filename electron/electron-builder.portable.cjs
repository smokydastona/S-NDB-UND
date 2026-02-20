// Dedicated electron-builder config for Windows "portable" builds.
// This avoids package.json's default NSIS target.

const pkg = require("./package.json");

const base = pkg.build || {};
const baseWin = base.win || {};

module.exports = {
  ...base,
  win: {
    ...baseWin,
    target: ["portable"],
    artifactName: "${productName}-Portable-${version}.${ext}",
  },
};
