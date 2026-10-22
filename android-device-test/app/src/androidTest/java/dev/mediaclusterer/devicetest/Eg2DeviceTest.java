package dev.mediaclusterer.devicetest;

import static org.junit.Assert.assertTrue;

import android.content.Context;
import android.content.Intent;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.util.Log;

import androidx.test.ext.junit.runners.AndroidJUnit4;
import androidx.test.platform.app.InstrumentationRegistry;
import androidx.test.uiautomator.By;
import androidx.test.uiautomator.UiDevice;
import androidx.test.uiautomator.UiObject2;
import androidx.test.uiautomator.Until;

import org.junit.Test;
import org.junit.runner.RunWith;

import java.util.regex.Pattern;

/**
 * Opens device-test.html in Chrome and waits for its "DONE" line. The page's
 * own log is read back through the accessibility tree and re-logged under the
 * EG2TEST tag, so the result is in logcat even if Chrome does not forward
 * console.log there.
 */
@RunWith(AndroidJUnit4.class)
public class Eg2DeviceTest {
    private static final String TAG = "EG2TEST";
    private static final String DEFAULT_BROWSER = "com.android.chrome";
    private static final String DEFAULT_URL =
            "https://claude-eg2-device-test.media-clusterer.pages.dev/device-test.html";
    private static final long TIMEOUT_MS = 20 * 60 * 1000;
    // Chrome first-run and promo buttons, across versions.
    private static final Pattern DISMISS = Pattern.compile(
            "(?i)(accept & continue|accept and continue|use without an account|continue as guest|"
                    + "no thanks|no, thanks|not now|skip|got it|ok|dismiss|allow)");

    @Test
    public void runEmbeddingGemma2() throws Exception {
        Bundle args = InstrumentationRegistry.getArguments();
        String url = args.getString("url", DEFAULT_URL);
        // Test Lab images can ship a Chrome too old for WebGPU; a sideloaded
        // Chromium (org.chromium.chrome) is picked with -e browser.
        String browser = args.getString("browser", DEFAULT_BROWSER);
        UiDevice device = UiDevice.getInstance(InstrumentationRegistry.getInstrumentation());
        Context ctx = InstrumentationRegistry.getInstrumentation().getContext();

        Log.i(TAG, "harness opening " + url + " in " + browser);
        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url))
                .setPackage(browser)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        ctx.startActivity(intent);
        device.wait(Until.hasObject(By.pkg(browser).depth(0)), 30_000);

        long deadline = SystemClock.uptimeMillis() + TIMEOUT_MS;
        int logged = 0;
        boolean done = false;
        boolean reopened = false;
        while (SystemClock.uptimeMillis() < deadline && !done) {
            dismissDialogs(device);
            // Chrome's first run can swallow the intent; send it again once.
            if (!reopened && device.hasObject(By.res(browser, "url_bar"))
                    && !device.hasObject(By.textContains("EG2"))
                    && !device.hasObject(By.textContains("START"))) {
                UiObject2 bar = device.findObject(By.res(browser, "url_bar"));
                if (bar != null && !String.valueOf(bar.getText()).contains("device-test")) {
                    Log.i(TAG, "harness re-sending intent after first run");
                    ctx.startActivity(intent);
                    reopened = true;
                }
            }
            String text = pageLog(device);
            if (text != null) {
                String[] lines = text.split("\n");
                for (int i = logged; i < lines.length; i++) Log.i(TAG, "page " + lines[i]);
                logged = Math.max(logged, lines.length);
                done = text.contains("DONE ");
            }
            if (!done) SystemClock.sleep(5_000);
        }
        String finalText = pageLog(device);
        Log.i(TAG, "harness finished done=" + done);
        assertTrue("page never reported DONE", done);
        assertTrue("page reported failures: " + finalText,
                finalText != null && finalText.contains("DONE pass"));
    }

    /** The page's <pre id="log"> as exposed to accessibility, or null. */
    private static String pageLog(UiDevice device) {
        UiObject2 node = device.findObject(By.textContains("START ua="));
        if (node == null) return null;
        try {
            return node.getText();
        } catch (RuntimeException stale) {
            return null;
        }
    }

    private static void dismissDialogs(UiDevice device) {
        for (UiObject2 b : device.findObjects(By.clickable(true).text(DISMISS))) {
            try {
                Log.i(TAG, "harness tapping '" + b.getText() + "'");
                b.click();
                SystemClock.sleep(1_000);
            } catch (RuntimeException stale) {
                // Gone between find and click.
            }
        }
    }
}
