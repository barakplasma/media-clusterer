import * as Sentry from "@sentry/browser";
import { captureConsoleIntegration } from "@sentry/browser";

// Read doNotTrack from persisted settings before the app initialises.
// This module is a side-effect import that runs at startup, so we read
// localStorage directly rather than going through the settings object.
const _saved = localStorage.getItem('mc_settings');
const _doNotTrack: boolean = _saved ? (JSON.parse(_saved).doNotTrack ?? false) : false;

Sentry.init({
  dsn: "https://06e453f266d043e6a12759fd3b778817@barakplasma.bugsink.com/3",
  release: __APP_VERSION__,
  enabled: !_doNotTrack,
  sendDefaultPii: false,
  integrations: [
    captureConsoleIntegration({ levels: ["error"] }),
  ],
});
