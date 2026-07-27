import 'vite/client'

declare module 'vite/client' {
  interface ImportMetaEnv {
    readonly VITE_UNSPLASH_ACCESS_KEY?: string,
  }
}
