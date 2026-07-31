import { expect, test } from "@playwright/test";
import { transitionRecordDetailNavigation } from "./record-detail-navigation";

test("controlled close and reopen clear linked-record navigation for the same root", () => {
  const linkedView = {
    previousOpen: true,
    viewOverride: {
      rootRecordId: "root-record",
      viewId: "linked-record",
    },
  };

  const closed = transitionRecordDetailNavigation(linkedView, false);
  expect(closed.viewOverride).toBeNull();

  const reopened = transitionRecordDetailNavigation(closed, true);
  expect(reopened.viewOverride).toBeNull();
});
