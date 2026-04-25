# Plan: EXIF and Video Metadata Extraction

## Context

The modal footer currently shows file `lastModified` timestamp, which is the filesystem modification date - not the actual capture date for photos or recording date for videos. User wants full EXIF metadata for photos and technical metadata for videos.

**Current Problem:**
- Photos show file copy/edit date, not actual capture date
- Videos show file modification date, not recording date
- No camera info (make, model, ISO, aperture, etc.)
- No video codec/bitrate information

**Desired Outcome:**
- Use actual capture/recording datetime for breadcrumbs and sorting
- Display camera settings (ISO, aperture, shutter speed, focal length)
- Display camera make/model
- Extract GPS coordinates if available
- Show video codec, bitrate, frame rate

## Implementation Approach

### Libraries to Add

1. **exifreader** (~20-50KB) - Photo EXIF/IPTC/XMP parsing
   - Supports JPEG, PNG, TIFF, HEIC, WebP, GIF, AVIF, JPEG XL
   - Install: `npm install exifreader`

2. **Native HTML5 Video API** - Basic video metadata (no library needed)
   - `video.duration`, `video.videoWidth`, `video.videoHeight`
   - Available from video element once loaded

3. **mediainfo.js** (~3-5MB WASM) - Optional for deep video metadata
   - Only if deeper video codec info needed beyond native API
   - Can be added later if required

### Files to Modify

1. **`src/types.ts`**
   - Add `ExifMetadata` interface
   - Add `VideoMetadata` interface
   - Extend `PhotoFile` with optional `exif` and `videoMeta` properties

2. **`src/app.ts`**
   - Add `extractExifMetadata()` function
   - Add `extractVideoMetadata()` function (native video API)
   - Integrate into `processFiles()` after file collection
   - Update `openFileModal()` to display metadata in modal footer
   - Use EXIF date for datetime breadcrumbs

3. **`package.json`**
   - Add `exifreader` dependency

4. **`index.html`**
   - Expand modal footer to show camera/video metadata
   - Keep layout compact with key info visible

### Data Flow

```
collectImages() → processFiles()
                           ↓
                    extractMetadataBatch()
                           ↓
              ┌──────────────┴──────────────┐
              ↓                              ↓
      extractExifMetadata()        extractVideoMetadata()
              ↓                              ↓
         Store in PhotoFile           Store in PhotoFile
              ↓                              ↓
         openFileModal()            openFileModal()
              ↓                              ↓
         Display in footer          Display in footer
```

### Key Design Decisions

**EXIF Date Fallback:**
- Primary: `DateTimeOriginal` (when photo was taken)
- Fallback: `DateTime` (when digitized)
- Last resort: `lastModified` (filesystem)

**Video Metadata Approach:**
- Use native HTML5 video API (no library overhead)
- Duration, resolution available from `<video>` element
- Extract when video loads in modal (lazy)

**Metadata Display:**
- Keep modal footer compact
- Show key metadata inline next to datetime
- Format: `2025/01/15 14:30 · Canon EOS R5 · ISO 400 · f/2.8 · 1/200s`

**Performance:**
- Batch EXIF extraction during `processFiles()` for photos
- Lazy video metadata extraction when opened in modal
- Cache extracted metadata in PhotoFile objects

### Implementation Steps

1. **Install exifreader**
   ```bash
   npm install exifreader
   ```

2. **Extend types** (`src/types.ts`)
   ```typescript
   interface ExifMetadata {
     dateTimeOriginal?: Date;
     make?: string;
     model?: string;
     iso?: number;
     exposureTime?: number;  // stored as reciprocal (1/200 = 0.005)
     fNumber?: number;
     focalLength?: number;
     gps?: { latitude?: number; longitude?: number };
   }

   interface VideoMetadata {
     duration?: number;
     width?: number;
     height?: number;
   }

   interface PhotoFile {
     // ... existing (name, size, lastModified, file, objectURL)
     exif?: ExifMetadata;
     videoMeta?: VideoMetadata;
   }
   ```

3. **Add EXIF extraction** (`src/app.ts`)
   ```typescript
   import ExifReader from 'exifreader';

   async function extractExifMetadata(file: File): Promise<ExifMetadata | null> {
     try {
       const tags = await ExifReader.load(file);
       const meta: ExifMetadata = {};
       if (tags['DateTimeOriginal']) {
         const parts = tags['DateTimeOriginal'].description.match(/(\d+):(\d+):(\d+)\s+(\d+):(\d+):(\d+)/);
         if (parts) meta.dateTimeOriginal = new Date(parts[1], parts[2]-1, parts[3], parts[4], parts[5], parts[6]);
       }
       meta.make = tags['Make']?.description;
       meta.model = tags['Model']?.description;
       meta.iso = tags['ISOSpeedRatings']?.value;
       meta.exposureTime = tags['ExposureTime']?.value;
       meta.fNumber = tags['FNumber']?.value;
       meta.focalLength = tags['FocalLength']?.value;
       if (tags.gps?.Latitude !== undefined) {
         meta.gps = { latitude: tags.gps.Latitude, longitude: tags.gps.Longitude };
       }
       return Object.keys(meta).length > 0 ? meta : null;
     } catch { return null; }
   }
   ```

4. **Extract video metadata** (`src/app.ts`)
   ```typescript
   function extractVideoMetadata(video: HTMLVideoElement): VideoMetadata {
     return {
       duration: video.duration,
       width: video.videoWidth,
       height: video.videoHeight
     };
   }
   ```

5. **Integrate into processFiles** (`src/app.ts`)
   - After collecting files, batch extract EXIF for images
   - Video metadata extracted lazily when modal opens

6. **Update modal footer** (`src/app.ts` - `openFileModal()`)
   - Show EXIF date for datetime breadcrumbs
   - Display camera settings inline: `Make Model · ISO · f/· 1/s`
   - For videos: duration and resolution

### Verification

1. Open a photo with EXIF → verify camera info shows in footer
2. Open a video → verify duration/resolution shows
3. Click datetime breadcrumb → filters by EXIF date, not file date
4. Test photo without EXIF → falls back to lastModified
5. Test HEIC files → exifreader supports them
