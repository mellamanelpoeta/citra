// Exporta filas a CSV y lo descarga. BOM para que Excel respete los acentos.
export function descargarCsv(nombre: string, filas: Record<string, string | number | boolean>[]) {
  if (!filas.length) return;
  const columnas = Object.keys(filas[0]);
  const celda = (v: string | number | boolean) => {
    const s = String(v);
    return /[",\n]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
  };
  const texto = [columnas.join(","), ...filas.map((f) => columnas.map((c) => celda(f[c])).join(","))].join("\n");
  const url = URL.createObjectURL(new Blob(["﻿" + texto], { type: "text/csv;charset=utf-8" }));
  const a = Object.assign(document.createElement("a"), { href: url, download: nombre });
  a.click();
  // Revocar en el mismo tick puede cancelar la descarga en Safari y Firefox.
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
