import { expect, test } from "@playwright/test";
import {
  canManageValidatedLocalPack,
  LOCAL_PACK_OPERATOR_COPY,
} from "./ValidatedLocalPackCard";

test("non-operators see status but not local-pack lifecycle controls", () => {
  expect(canManageValidatedLocalPack({ can_manage_pack: false })).toBe(false);
  expect(LOCAL_PACK_OPERATOR_COPY).toBe(
    "This model pack is managed by the machine operator. You can review its status here."
  );
});

test("operators may use local-pack lifecycle controls", () => {
  expect(canManageValidatedLocalPack({ can_manage_pack: true })).toBe(true);
});
