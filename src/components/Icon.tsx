/** Thin wrapper over Google Material Symbols (loaded in index.html). */
export function Icon({
  name,
  className = "",
  fill = false,
  size,
  style,
}: {
  name: string;
  className?: string;
  fill?: boolean;
  size?: number;
  style?: React.CSSProperties;
}) {
  return (
    <span
      className={`material-symbols-outlined${fill ? " fill" : ""} ${className}`}
      style={size ? { fontSize: size, ...style } : style}
      aria-hidden="true"
    >
      {name}
    </span>
  );
}
