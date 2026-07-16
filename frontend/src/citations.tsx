import type { ReactNode } from "react";
import ReactMarkdown from "react-markdown";

// A brief cites a paper by its arXiv id in brackets, e.g. [2512.24565]. Ids are new-style only
// (the corpus is the last ~12 months), and brackets are never used for anything else, so this
// one pattern catches every citation. A fresh regex per call keeps the global lastIndex from
// leaking between matchAll and replace.
const PATTERN = "\\[(\\d{4}\\.\\d{4,5}(?:v\\d+)?)\\]";
const cite = () => new RegExp(PATTERN, "g");

export const arxivUrl = (id: string) => `https://arxiv.org/abs/${id}`;

const idFromUrl = (url: string): string | null => {
  const m = url.match(/\/abs\/(\d{4}\.\d{4,5}(?:v\d+)?)$/);
  return m ? m[1] : null;
};

// The end-of-animation cleanup, as a stable reference so a prior one can be removed before the
// next flash restarts it. Pulling the class after the fade keeps the card's normal styling.
function clearFlash(e: Event): void {
  (e.currentTarget as HTMLElement).classList.remove("cite-flash");
}

// Scroll a paper's on-page entry fully into view and highlight it so it is obvious which one was
// selected. The highlight fades itself out (CSS) and is torn down when the animation ends.
// Callers resolve the element within their own subtree: both tab workspaces stay mounted, so a
// document-wide lookup could land on the hidden one.
export function flashPaper(el: Element | null | undefined): void {
  if (!(el instanceof HTMLElement)) return;
  const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  el.scrollIntoView({ behavior: reduce ? "auto" : "smooth", block: "center" });
  // Restart cleanly even if a prior flash is still running: drop the class, detach its pending
  // end-listener, force a reflow so the browser sees a fresh start, then re-apply.
  el.classList.remove("cite-flash");
  el.removeEventListener("animationend", clearFlash);
  void el.offsetWidth;
  el.classList.add("cite-flash");
  el.addEventListener("animationend", clearFlash, { once: true });
}

function CiteLink({
  id,
  label,
  onCite,
}: {
  id: string;
  label: string;
  onCite?: (id: string) => void;
}) {
  // With a handler the click scrolls to the paper on-page; without one it opens arXiv. The
  // arXiv href stays as the modifier-click / no-JS fallback either way.
  return (
    <a
      href={arxivUrl(id)}
      target={onCite ? undefined : "_blank"}
      rel="noreferrer"
      onClick={
        onCite
          ? (e) => {
              e.preventDefault();
              onCite(id);
            }
          : undefined
      }
    >
      {label}
    </a>
  );
}

// For plain-text spots (key ideas, open problems, direction summaries): return the text with
// each citation turned into a clickable link, leaving the [id] itself visible.
export function linkifyCitations(text: string, onCite?: (id: string) => void): ReactNode {
  const parts: ReactNode[] = [];
  let last = 0;
  for (const m of text.matchAll(cite())) {
    const at = m.index ?? 0;
    if (at > last) parts.push(text.slice(last, at));
    parts.push(<CiteLink key={at} id={m[1]} label={m[0]} onCite={onCite} />);
    last = at + m[0].length;
  }
  if (parts.length === 0) return text;
  if (last < text.length) parts.push(text.slice(last));
  return parts;
}

// For markdown briefs: rewrite each [id] to a link whose visible text keeps the brackets, then
// let react-markdown render it. The [id] renders exactly as before, only now it is clickable.
export function CitedMarkdown({
  children,
  onCite,
}: {
  children: string;
  onCite?: (id: string) => void;
}) {
  const linked = children.replace(cite(), (_m, id: string) => `[\\[${id}\\]](${arxivUrl(id)})`);
  return (
    <ReactMarkdown
      components={{
        a: ({ href, children: kids }) => {
          const id = href ? idFromUrl(href) : null;
          if (onCite && id) {
            return (
              <a
                href={href}
                rel="noreferrer"
                onClick={(e) => {
                  e.preventDefault();
                  onCite(id);
                }}
              >
                {kids}
              </a>
            );
          }
          return (
            <a href={href} target="_blank" rel="noreferrer">
              {kids}
            </a>
          );
        },
      }}
    >
      {linked}
    </ReactMarkdown>
  );
}
