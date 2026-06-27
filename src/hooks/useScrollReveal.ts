import { useEffect } from "react";

/**
 * Reveals elements as they scroll into view.
 *
 * Mark any element with `data-reveal` and it fades/rises in the first time it
 * enters the viewport. Use `data-reveal="fade"` for an opacity-only variant
 * (table rows, where transforms are flaky in some WebViews). Stagger siblings
 * by setting `style={{ "--reveal-delay": "120ms" }}`.
 *
 * One observer per mount handles the whole scroll container, and a
 * MutationObserver picks up nodes added after async data loads (fetched lists,
 * model cards, …). Honors `prefers-reduced-motion` via the CSS in index.css —
 * the class still gets added, it just snaps in without transition.
 */
export function useScrollReveal(rootSelector = "#doclab-main") {
  useEffect(() => {
    const root = document.querySelector(rootSelector);
    if (!root || typeof IntersectionObserver === "undefined") return;

    const io = new IntersectionObserver(
      (entries) => {
        for (const entry of entries) {
          if (entry.isIntersecting) {
            entry.target.classList.add("is-revealed");
            io.unobserve(entry.target);
          }
        }
      },
      // Trigger a touch before the element is fully in view so it lands settled.
      { root: null, rootMargin: "0px 0px -8% 0px", threshold: 0.05 },
    );

    const observeWithin = (scope: ParentNode) => {
      scope
        .querySelectorAll<HTMLElement>("[data-reveal]:not(.is-revealed)")
        .forEach((el) => io.observe(el));
    };

    observeWithin(root);

    // Content that renders after a fetch won't exist on first pass — watch for it.
    const mo = new MutationObserver((mutations) => {
      for (const m of mutations) {
        m.addedNodes.forEach((node) => {
          if (!(node instanceof HTMLElement)) return;
          if (node.matches("[data-reveal]:not(.is-revealed)")) io.observe(node);
          observeWithin(node);
        });
      }
    });
    mo.observe(root, { childList: true, subtree: true });

    return () => {
      io.disconnect();
      mo.disconnect();
    };
  }, [rootSelector]);
}
