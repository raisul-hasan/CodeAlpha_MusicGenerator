"""Verify a running studio with a separate headless browser.

Start the app first. Then run:
    python scripts/browser_check.py --url http://127.0.0.1:8501

Uses installed Microsoft Edge by default. Use --channel chromium after running
`python -m playwright install chromium` on a computer without Edge.
"""
import argparse
import json
import sys
import time
import urllib.error
import urllib.request
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mido
from playwright.sync_api import sync_playwright

from musicgen.io import write_json
from musicgen.paths import REPORTS
from musicgen.midi import read_midi


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8501")
    parser.add_argument("--channel", default="msedge")
    args = parser.parse_args()
    REPORTS.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + 25
    while True:
        try:
            with urllib.request.urlopen(args.url.rstrip("/") + "/_stcore/health", timeout=2) as response:
                if response.status == 200:
                    break
        except (OSError, urllib.error.URLError):
            pass
        if time.monotonic() >= deadline:
            raise RuntimeError("Studio is not ready. Start it with start.ps1 before this check.")
        time.sleep(.25)
    checks = []
    with sync_playwright() as playwright:
        browser = playwright.chromium.launch(channel=args.channel, headless=True)
        context = browser.new_context(viewport={"width": 1440, "height": 1100}, accept_downloads=True)
        page = context.new_page()
        page.goto(args.url, wait_until="domcontentloaded")
        page.get_by_role("heading", name="Make room for a new melody.").wait_for(timeout=45000)
        page.get_by_text("Featured composition", exact=True).wait_for(timeout=30000)
        page.locator("audio").first.wait_for()
        audio = page.locator("audio").first
        audio.evaluate("element => element.play()")
        page.wait_for_function("document.querySelector('audio').currentTime > 0", timeout=15000)
        audio.evaluate("element => element.pause()")
        checks.append("Featured WAV loads and plays in the browser")
        page.get_by_role("button", name="Generate composition", exact=False).click()
        page.get_by_text("Your composition is ready. Press play to listen.", exact=True).wait_for(timeout=30000)
        page.get_by_role("heading", name="Your new composition", exact=True).wait_for()
        checks.append("Generate button creates a new composition")
        with page.expect_download() as download:
            page.get_by_role("button", name="Download MIDI", exact=True).click()
        midi_download = download.value
        midi_path = REPORTS / "browser_download.mid"
        midi_download.save_as(midi_path)
        midi = mido.MidiFile(midi_path)
        note_count = sum(message.type == "note_on" and message.velocity > 0
                         for track in midi.tracks for message in track)
        assert note_count == 128, note_count
        onsets = sorted({note.start for note in read_midi(midi_path)})
        assert all(right - left >= .249 for left, right in zip(onsets, onsets[1:])), "The app served outdated rhythm rules."
        checks.append("MIDI download is a parseable 128-note composition")
        with page.expect_download() as download:
            page.get_by_role("button", name="Download WAV", exact=True).click()
        wav_path = REPORTS / "browser_download.wav"
        download.value.save_as(wav_path)
        with wave.open(str(wav_path), "rb") as wav:
            assert wav.getnframes() > 0
            assert wav.getframerate() == 22050
        checks.append("WAV download has valid audio frames and sample rate")
        page.screenshot(path=str(REPORTS / "studio_desktop.png"), full_page=True)
        page.get_by_role("tab", name="Example library", exact=True).click()
        page.get_by_role("heading", name="A few places to begin", exact=True).wait_for()
        assert page.locator("audio:visible").count() == 3
        checks.append("Example library displays three playable compositions")
        page.get_by_role("tab", name="About the model", exact=True).click()
        page.get_by_role("heading", name="Learned from real piano performances", exact=True).wait_for()
        page.get_by_text("Recorded training and validation results.", exact=True).wait_for()
        checks.append("Model information displays the recorded evaluation and learning curves")
        page.get_by_role("tab", name="Compose", exact=True).click()
        page.get_by_role("heading", name="Your new composition", exact=True).wait_for()
        page.set_viewport_size({"width": 390, "height": 844})
        page.screenshot(path=str(REPORTS / "studio_mobile.png"), full_page=True)
        assert page.get_by_role("button", name="Generate composition", exact=False).is_visible()
        page.get_by_role("heading", name="Your new composition", exact=True).scroll_into_view_if_needed()
        page.screenshot(path=str(REPORTS / "studio_mobile_player.png"), full_page=True)
        checks.append("Compose controls remain available at mobile width")
        assert page.locator('[data-testid="stException"]').count() == 0
        browser.close()
    write_json(REPORTS / "browser_verification.json", {"passed": True, "url": args.url,
                                                       "browser": args.channel, "checks": checks})
    print(json.dumps({"passed": True, "checks": checks}, indent=2))


if __name__ == "__main__":
    main()
