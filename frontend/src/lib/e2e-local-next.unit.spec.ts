import { expect, test } from "@playwright/test";
import { spawnSync } from "node:child_process";
import { readFileSync } from "node:fs";
import { resolve } from "node:path";

import {
  localOnlyNextServerCommand,
  MACOS_SANDBOX_PROFILE,
} from "../../e2e/helpers/local-only-next";

test.describe("localOnlyNextServerCommand", () => {
  test("pins the patched Next.js runtime and transitive processors", () => {
    const packageJson = JSON.parse(
      readFileSync(resolve(process.cwd(), "package.json"), "utf-8")
    ) as {
      dependencies: Record<string, string>;
      devDependencies: Record<string, string>;
      overrides?: Record<string, string>;
    };

    expect(packageJson.dependencies.next).toBe("16.2.12");
    expect(packageJson.devDependencies["eslint-config-next"]).toBe("16.2.12");
    expect(packageJson.overrides).toMatchObject({
      postcss: "8.5.24",
      sharp: "0.35.3",
    });
  });

  test("bundles the documented fonts without next/font network resolution", () => {
    const layout = readFileSync(
      resolve(process.cwd(), "src/app/layout.tsx"),
      "utf-8"
    );
    const packageJson = JSON.parse(
      readFileSync(resolve(process.cwd(), "package.json"), "utf-8")
    ) as { dependencies: Record<string, string> };

    expect(layout).not.toContain("next/font/google");
    expect(layout).toContain('@fontsource/source-serif-4/latin-400.css');
    expect(layout).toContain(
      '@fontsource/source-serif-4/latin-400-italic.css'
    );
    expect(layout).toContain('@fontsource/source-sans-3/latin-400.css');
    expect(layout).toContain('@fontsource/ibm-plex-mono/latin-400.css');
    expect(packageJson.dependencies).toMatchObject({
      "@fontsource/ibm-plex-mono": expect.any(String),
      "@fontsource/source-sans-3": expect.any(String),
      "@fontsource/source-serif-4": expect.any(String),
    });
  });

  test("runs Next.js inside a macOS loopback-only OS sandbox", () => {
    const command = localOnlyNextServerCommand("darwin");

    expect(command).toContain("/usr/bin/sandbox-exec");
    expect(command).toContain("(deny network-outbound)");
    expect(command).toContain(
      '(allow network-outbound (remote ip "localhost:*"))'
    );
    expect(command).toMatch(/npm run dev -- --hostname 127\.0\.0\.1$/);
  });

  test("fails closed when an OS network sandbox is unavailable", () => {
    expect(() => localOnlyNextServerCommand("linux")).toThrow(
      /supported OS network sandbox/i
    );
  });

  test("the OS profile allows loopback but blocks an external socket", () => {
    test.skip(process.platform !== "darwin", "macOS sandbox regression");
    const probe = [
      "const net = require('node:net');",
      "const loopback = net.connect({host: '127.0.0.1', port: 9});",
      "loopback.on('connect', () => loopback.destroy());",
      "loopback.on('error', (error) => {",
      "  if (error.code === 'EPERM') process.exit(11);",
      "  const external = net.connect({host: '1.1.1.1', port: 53});",
      "  external.on('connect', () => process.exit(12));",
      "  external.on('error', (externalError) => {",
      "    process.exit(externalError.code === 'EPERM' ? 0 : 13);",
      "  });",
      "});",
      "setTimeout(() => process.exit(14), 2000);",
    ].join("\n");

    const result = spawnSync(
      "/usr/bin/sandbox-exec",
      ["-p", MACOS_SANDBOX_PROFILE, process.execPath, "-e", probe],
      { encoding: "utf-8", timeout: 5_000 }
    );

    expect(result.status, result.stderr).toBe(0);
  });
});
