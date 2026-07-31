import { expect, test } from "@playwright/test";

import { fmtDay, fmtShort, yearOf } from "@/lib/format-date";

test("clinical calendar dates are read from the source day without timezone drift", () => {
  expect(fmtDay("2019-01-01T00:00:00Z")).toBe("Jan 1, 2019");
  expect(fmtShort("2019-01-01T00:00:00Z")).toBe("Jan 1 '19");
  expect(yearOf("2019-01-01T00:00:00Z")).toBe(2019);
});
