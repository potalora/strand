export interface RecordDetailViewOverride {
  rootRecordId: string;
  viewId: string;
}

export interface RecordDetailNavigation {
  previousOpen: boolean;
  viewOverride: RecordDetailViewOverride | null;
}

/** Clears linked-record navigation whenever the controlled Sheet open prop changes. */
export function transitionRecordDetailNavigation(
  state: RecordDetailNavigation,
  open: boolean,
): RecordDetailNavigation {
  if (state.previousOpen === open) return state;
  return { previousOpen: open, viewOverride: null };
}
