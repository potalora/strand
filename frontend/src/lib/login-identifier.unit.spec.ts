import { expect, test } from "@playwright/test";
import { maskLoginIdentifier } from "./login-identifier";

test("masks account names by Unicode code point", () => {
  expect(maskLoginIdentifier("a")).toBe("•");
  expect(maskLoginIdentifier("ab")).toBe("••");
  expect(maskLoginIdentifier("alice")).toBe("al•••");
  expect(maskLoginIdentifier("😀xray")).toBe("😀x•••");
});
