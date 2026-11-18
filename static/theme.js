"use strict";
// Apply the saved theme before first paint so there is no flash. Loaded as a
// blocking script in <head>; kept in a file (not inline) so the
// Content-Security-Policy can forbid inline scripts entirely.
try {
  const saved = localStorage.getItem("theme") || "system";
  const dark = saved === "dark" || (saved === "system" && matchMedia("(prefers-color-scheme: dark)").matches);
  document.documentElement.dataset.theme = dark ? "dark" : "light";
} catch (_) { document.documentElement.dataset.theme = "dark"; }
