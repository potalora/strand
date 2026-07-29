export const MACOS_SANDBOX_PROFILE = [
  "(version 1)",
  "(allow default)",
  "(deny network-outbound)",
  '(allow network-outbound (remote ip "localhost:*"))',
].join(" ");

function shellQuote(value: string): string {
  return `'${value.replaceAll("'", "'\\''")}'`;
}

export function localOnlyNextServerCommand(
  platform: NodeJS.Platform = process.platform
): string {
  if (platform !== "darwin") {
    throw new Error(
      "E2E_LOCAL_ONLY requires a supported OS network sandbox for the Next.js server."
    );
  }
  return [
    "/usr/bin/sandbox-exec",
    "-p",
    shellQuote(MACOS_SANDBOX_PROFILE),
    "npm",
    "run",
    "dev",
    "--",
    "--hostname",
    "127.0.0.1",
  ].join(" ");
}
