import { useEffect, useState } from "react";
import { AssistantView } from "./components/AssistantView";
import { QualityPage } from "./components/QualityPage";

type View = "assistant" | "about";

const ABOUT_HASH = "#/acerca";
/** The page's former address, kept so existing links still land on it. */
const LEGACY_ABOUT_HASH = "#/calidad";

/** Map the URL hash to a view, so the about page is directly linkable. */
function viewFromHash(): View {
  const { hash } = window.location;
  return hash === ABOUT_HASH || hash === LEGACY_ABOUT_HASH ? "about" : "assistant";
}

export function App() {
  const [view, setView] = useState<View>(viewFromHash);

  useEffect(() => {
    const onHashChange = () => setView(viewFromHash());
    window.addEventListener("hashchange", onHashChange);
    return () => window.removeEventListener("hashchange", onHashChange);
  }, []);

  function go(next: View) {
    window.location.hash = next === "about" ? ABOUT_HASH : "";
  }

  return (
    <div className="app">
      <header className="app-header">
        <h1 className="app-title">BOE RAG</h1>
        <p className="app-subtitle">
          Respuestas <strong>citadas y verificables</strong> sobre legislación española,
          fundamentadas en el Boletín Oficial del Estado.
        </p>
        <nav className="app-nav" aria-label="Secciones">
          <button
            type="button"
            className={`nav-tab ${view === "assistant" ? "active" : ""}`}
            aria-current={view === "assistant" ? "page" : undefined}
            onClick={() => go("assistant")}
          >
            Asistente
          </button>
          <button
            type="button"
            className={`nav-tab ${view === "about" ? "active" : ""}`}
            aria-current={view === "about" ? "page" : undefined}
            onClick={() => go("about")}
          >
            Acerca de
          </button>
        </nav>
      </header>

      <main className="app-main">
        {view === "assistant" ? <AssistantView /> : <QualityPage />}
      </main>

      <footer className="app-footer">
        <p>
          Demo de portafolio. Las respuestas pueden contener errores: verifica siempre
          con la fuente oficial enlazada. No constituye asesoramiento legal.
        </p>
      </footer>
    </div>
  );
}
