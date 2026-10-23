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

import java.io.ByteArrayOutputStream;
import java.util.LinkedHashSet;
import java.util.Set;
import java.util.regex.Matcher;
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
    private static final long TIMEOUT_MS = 40 * 60 * 1000;
    // Chrome first-run and promo buttons, across versions.
    private static final Pattern DISMISS = Pattern.compile(
            "(?i)(accept & continue|accept and continue|use without an account|continue as guest|stay signed out|"
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

        Intent intent = new Intent(Intent.ACTION_VIEW, Uri.parse(url))
                .setPackage(browser)
                .addFlags(Intent.FLAG_ACTIVITY_NEW_TASK | Intent.FLAG_ACTIVITY_CLEAR_TASK);
        // Some GPUs (Pixel 10's PowerVR) are on Chrome's WebGPU blocklist;
        // -e flag enable-unsafe-webgpu switches it on in chrome://flags first.
        String flag = args.getString("flag", "");
        if (!flag.isEmpty()) {
            ctx.startActivity(intent);
            device.wait(Until.hasObject(By.pkg(browser).depth(0)), 30_000);
            enableFlag(device, browser, flag);
        }

        Log.i(TAG, "harness opening " + url + " in " + browser);
        ctx.startActivity(intent);
        device.wait(Until.hasObject(By.pkg(browser).depth(0)), 30_000);

        long deadline = SystemClock.uptimeMillis() + TIMEOUT_MS;
        int logged = 0;
        boolean done = false;
        boolean reopened = false;
        Set<String> seenTexts = new LinkedHashSet<>();
        boolean sampled = false;
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
            } else {
                // The selector found nothing (the full app page): read every
                // text and content-desc in the window dump instead.
                done = scanHierarchy(device, seenTexts, !sampled);
                sampled = true;
            }
            if (!done) SystemClock.sleep(5_000);
        }
        String finalText = pageLog(device);
        if (finalText == null) finalText = String.join("\n", seenTexts);
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

    private static final Pattern ATTR = Pattern.compile("(?:text|content-desc)=\"([^\"]*)\"");

    /** Log new strings from the window dump; true once one contains "DONE ". */
    private static boolean scanHierarchy(UiDevice device, Set<String> seen, boolean sample) {
        String xml;
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            device.dumpWindowHierarchy(out);
            xml = out.toString("UTF-8");
        } catch (Exception e) {
            Log.i(TAG, "harness dump failed " + e);
            return false;
        }
        if (sample) {
            Log.i(TAG, "harness dump " + xml.length() + " chars");
            for (int i = 0; i < Math.min(xml.length(), 12_000); i += 3_000)
                Log.i(TAG, "harness dump[" + i + "] " + xml.substring(i, Math.min(xml.length(), i + 3_000)));
        }
        boolean done = false;
        Matcher m = ATTR.matcher(xml);
        while (m.find()) {
            String v = m.group(1).replace("&#10;", "\n").replace("&quot;", "\"").replace("&lt;", "<")
                    .replace("&gt;", ">").replace("&apos;", "'").replace("&amp;", "&");
            if (v.isEmpty()) continue;
            for (String line : v.split("\n")) {
                if (line.isEmpty() || !seen.add(line)) continue;
                Log.i(TAG, "page " + line);
                if (line.startsWith("DONE ")) done = true;
            }
        }
        return done;
    }

    private static final Pattern NODE = Pattern.compile(
            "text=\"([^\"]*)\" resource-id=\"([^\"]*)\" class=\"([^\"]*)\" package=\"([^\"]*)\" content-desc=\"([^\"]*)\".*?bounds=\"([^\"]*)\"");

    /** One compact line per non-empty node of the browser window. */
    private static void logNodes(UiDevice device, String pkg) {
        try {
            ByteArrayOutputStream out = new ByteArrayOutputStream();
            device.dumpWindowHierarchy(out);
            Matcher m = NODE.matcher(out.toString("UTF-8"));
            int n = 0;
            while (m.find() && n < 150) {
                if (!m.group(4).equals(pkg) || (m.group(1).isEmpty() && m.group(5).isEmpty() && m.group(2).isEmpty())) continue;
                Log.i(TAG, "harness node " + m.group(3).replace("android.widget.", "") + " id=" + m.group(2).replace(pkg + ":id/", "")
                        + " text='" + m.group(1) + "' desc='" + m.group(5) + "' " + m.group(6));
                n++;
            }
        } catch (Exception e) {
            Log.i(TAG, "harness node dump failed " + e);
        }
    }

    /** Set chrome://flags/#<flag> to Enabled and relaunch the browser. */
    private static void enableFlag(UiDevice device, String browser, String flag) {
        for (int i = 0; i < 4; i++) {
            dismissDialogs(device);
            SystemClock.sleep(2_000);
        }
        UiObject2 bar = device.wait(Until.findObject(By.res(browser, "url_bar")), 20_000);
        if (bar == null) bar = device.findObject(By.res(browser, "search_box_text"));
        if (bar == null) {
            Log.i(TAG, "harness flag: no url bar");
            scanHierarchy(device, new LinkedHashSet<>(), true);
            return;
        }
        bar.click();
        SystemClock.sleep(1_000);
        UiObject2 edit = device.wait(Until.findObject(By.res(browser, "url_bar")), 10_000);
        // The #anchor scrolls the page to this flag and highlights it.
        String flagsUrl = "chrome://flags/#" + flag;
        edit.setText(flagsUrl);
        // On the Pixel 10 the Enter key only opened the suggestion list; tap
        // the navigation suggestion when there is one.
        UiObject2 suggestion = device.wait(Until.findObject(By.res(browser, "line_1").text(flagsUrl)), 3_000);
        if (suggestion != null) suggestion.click();
        else device.pressEnter();
        SystemClock.sleep(6_000);
        // The flag's <select> is a plain View whose text is its current value,
        // just below the "<flag>_name" title.
        UiObject2 name = device.wait(Until.findObject(By.res(flag + "_name")), 15_000);
        UiObject2 dropdown = null;
        if (name != null) {
            int top = name.getVisibleBounds().top;
            for (UiObject2 v : device.findObjects(By.pkg(browser).text(Pattern.compile("Default|Enabled|Disabled")))) {
                int t = v.getVisibleBounds().top;
                if (t > top && (dropdown == null || t < dropdown.getVisibleBounds().top)) dropdown = v;
            }
        }
        Log.i(TAG, "harness flag: name=" + (name != null) + " value=" + (dropdown == null ? "none" : dropdown.getText()));
        if (dropdown == null) {
            Log.i(TAG, "harness flag: no dropdown");
            logNodes(device, browser);
            return;
        }
        dropdown.click();
        UiObject2 enabled = device.wait(Until.findObject(By.text("Enabled")), 10_000);
        if (enabled == null) {
            Log.i(TAG, "harness flag: no Enabled option");
            return;
        }
        enabled.click();
        UiObject2 relaunch = device.wait(Until.findObject(By.text(Pattern.compile("(?i)relaunch"))), 10_000);
        Log.i(TAG, "harness flag " + flag + " enabled, relaunch=" + (relaunch != null));
        if (relaunch != null) relaunch.click();
        else logNodes(device, browser);
        SystemClock.sleep(10_000);
    }

    // First-run buttons by id, whatever this Chrome/Chromium build labels them.
    private static final Pattern FRE_IDS = Pattern.compile(
            ".*:id/(signin_fre_dismiss_button|terms_accept|negative_button|button_secondary|ack_button)");

    private static void dismissDialogs(UiDevice device) {
        for (UiObject2 b : device.findObjects(By.res(FRE_IDS))) {
            try {
                Log.i(TAG, "harness tapping id " + b.getResourceName() + " '" + b.getText() + "'");
                b.click();
                SystemClock.sleep(1_000);
            } catch (RuntimeException stale) {
                // Gone between find and click.
            }
        }
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
