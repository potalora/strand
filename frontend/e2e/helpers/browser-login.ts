import type { Page } from "@playwright/test";

/**
 * Login via browser UI. Retries on rate limit (429).
 */
export async function browserLogin(
  page: Page,
  loginIdentifier: string,
  password: string
): Promise<void> {
  for (let attempt = 0; attempt < 8; attempt++) {
    await page.goto("/login");
    await page.locator("#loginIdentifier").fill(loginIdentifier);
    await page.locator("#password").fill(password);

    // Listen for potential rate-limit response
    const responsePromise = page.waitForResponse(
      (res) => res.url().includes("/auth/login"),
      { timeout: 10_000 }
    ).catch(() => null);

    await page.locator('button[type="submit"]').click();

    const response = await responsePromise;
    if (response && response.status() === 429) {
      // Rate limited — wait and retry
      const waitMs = Math.min(3000 * Math.pow(2, attempt), 15_000);
      await page.waitForTimeout(waitMs);
      continue;
    }

    await page.waitForURL(/\/$/, { timeout: 30_000 });
    return;
  }
  throw new Error("browserLogin: exceeded rate limit retries");
}
