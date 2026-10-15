import { createContext, useContext } from "react";
import type { AppData } from ".";

export const DataContext = createContext<AppData | null>(null);

export function useData(): AppData {
  const d = useContext(DataContext);
  if (!d) throw new Error("useData debe usarse dentro de DataContext");
  return d;
}
