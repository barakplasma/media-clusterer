import * as Sentry from "@sentry/browser";
import { captureConsoleIntegration } from "@sentry/browser";

Sentry.init({
  dsn: "https://06e453f266d043e6a12759fd3b778817@barakplasma.bugsink.com/3",
  release: __APP_VERSION__,
  sendDefaultPii: false,
  integrations: [
    captureConsoleIntegration({ levels: ["error"] }),
  ],
});
