export function maskLoginIdentifier(value: string): string {
  const codePoints = Array.from(value);
  if (codePoints.length <= 2) return "•".repeat(codePoints.length);
  return `${codePoints.slice(0, 2).join("")}•••`;
}
