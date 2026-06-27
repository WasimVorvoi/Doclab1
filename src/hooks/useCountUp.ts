import { useEffect, useRef, useState } from "react";

const prefersReducedMotion =
  typeof window !== "undefined" &&
  window.matchMedia?.("(prefers-reduced-motion: reduce)").matches;

/**
 * Animate a number from 0 → `target` once, on mount.
 * Returns the live value formatted to `decimals` places.
 */
export function useCountUp(
  target: number,
  { decimals = 0, duration = 1100, delay = 0 }: {
    decimals?: number;
    duration?: number;
    delay?: number;
  } = {},
): string {
  const [value, setValue] = useState(prefersReducedMotion ? target : 0);
  const frame = useRef<number>(0);

  useEffect(() => {
    if (prefersReducedMotion) return;
    let start: number | null = null;
    const startTimer = window.setTimeout(() => {
      const tick = (now: number) => {
        if (start === null) start = now;
        const t = Math.min(1, (now - start) / duration);
        // easeOutExpo for a quick, settling count
        const eased = t === 1 ? 1 : 1 - Math.pow(2, -10 * t);
        setValue(target * eased);
        if (t < 1) frame.current = requestAnimationFrame(tick);
      };
      frame.current = requestAnimationFrame(tick);
    }, delay);

    return () => {
      window.clearTimeout(startTimer);
      cancelAnimationFrame(frame.current);
    };
  }, [target, duration, delay]);

  return value.toFixed(decimals);
}
