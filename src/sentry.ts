import * as Sentry from '@sentry/browser'
import { captureConsoleIntegration } from '@sentry/browser'

// Read doNotTrack from persisted settings before the app initialises.
// This module is a side-effect import that runs at startup, so we read
// localStorage directly rather than going through the settings object.
const _saved = localStorage.getItem('mc_settings')
const _doNotTrack: boolean = _saved ? (JSON.parse(_saved).doNotTrack ?? false) : false

// Only real deployments report. Local dev servers, e2e runs and branch previews
// (<branch>.media-clusterer.pages.dev) trip deliberate failures such as the
// truncated-video fixture and filled Bugsink with them.
const _prodHosts = ['barakplasma.github.io', 'media-clusterer.pages.dev']

Sentry.init({
  dsn: 'https://06e453f266d043e6a12759fd3b778817@barakplasma.bugsink.com/3',
  release: __APP_VERSION__,
  enabled: !_doNotTrack && _prodHosts.includes(location.hostname),
  sendDefaultPii: false,
  integrations: [captureConsoleIntegration({ levels: ['error'] })]
})
