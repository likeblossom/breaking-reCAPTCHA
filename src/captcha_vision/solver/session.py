"""
Core reCAPTCHA solving logic.

Challenge taxonomy (same three types handled by aplesner/Breaking-reCAPTCHAv2):
  3x3_static   — Select all images matching X (9 tiles, one-shot).
  3x3_dynamic  — Select all images matching X; clicked tiles refresh with new
                 images (identified by "none" in the wrapper text).
  4x4          — 4×4 grid of tiles from a single large image ("squares" in
                 wrapper text).  We classify each tile independently since
                 we have no segmentation model.
"""
from __future__ import annotations

import csv
import logging
import random
import time
from datetime import datetime
from pathlib import Path

from playwright.sync_api import Frame, Page

from captcha_vision.inference.predictor import Decision, Predictor

from .browser import get_challenge_frame, get_checkbox_frame, human_click, human_move

log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Challenge-text → model class name
# ---------------------------------------------------------------------------

# Keys are substrings of the reCAPTCHA challenge prompt (lowercase).
# Longest match wins so "traffic lights" beats "traffic".
_LABEL_MAP: dict[str, str] = {
    "bicycle": "Bicycle",
    "bicycles": "Bicycle",
    "bridge": "Bridge",
    "bridges": "Bridge",
    "bus": "Bus",
    "buses": "Bus",
    "car": "Car",
    "cars": "Car",
    "vehicles": "Car",
    "chimney": "Chimney",
    "chimneys": "Chimney",
    "crosswalk": "Crosswalk",
    "crosswalks": "Crosswalk",
    "fire hydrant": "Hydrant",
    "fire hydrants": "Hydrant",
    "hydrant": "Hydrant",
    "hydrants": "Hydrant",
    "motorcycle": "Motorcycle",
    "motorcycles": "Motorcycle",
    "mountain": "Mountain",
    "mountains": "Mountain",
    "palm tree": "Palm",
    "palm trees": "Palm",
    "palm": "Palm",
    "staircase": "Stairs",
    "stairs": "Stairs",
    "stair": "Stairs",
    "tractor": "Tractor",
    "tractors": "Tractor",
    "traffic lights": "Traffic Light",
    "traffic light": "Traffic Light",
    "other": "Other",
}


def _parse_target(text: str) -> str | None:
    """
    Find the best matching class for the challenge prompt text.
    Returns the model class name (e.g. "Traffic Light") or None.
    """
    lower = text.lower()
    best: tuple[int, str] | None = None
    for key, cls in _LABEL_MAP.items():
        if key in lower:
            if best is None or len(key) > best[0]:
                best = (len(key), cls)
    return best[1] if best else None


# ---------------------------------------------------------------------------
# SolverSession
# ---------------------------------------------------------------------------

class SolverSession:
    """
    Drives reCAPTCHA challenges using a trained Predictor.

    Usage
    -----
    ::

        from playwright.sync_api import sync_playwright
        from captcha_vision.inference.predictor import Predictor
        from captcha_vision.solver import SolverSession

        predictor = Predictor("models/recaptcha_tiles_run1/best_model.pt", threshold=0.80)
        session   = SolverSession(predictor)

        with sync_playwright() as pw:
            ctx, page = launch(pw)
            solved = session.run(page, "https://www.google.com/recaptcha/api2/demo")
            ctx.close()

    Parameters
    ----------
    predictor:
        Loaded Predictor (checkpoint + threshold already configured).
    save_dir:
        Directory for tile PNG screenshots and the session CSV log.
    verbose:
        Print per-tile classification results to stdout.
    """

    def __init__(
        self,
        predictor: Predictor,
        *,
        save_dir: str | Path = "artifacts/solver_logs",
        verbose: bool = True,
        threshold_4x4: float = 0.35,
        save_tiles: bool = False,
        top_n: int = 3,
    ) -> None:
        self.predictor = predictor
        self.threshold_4x4 = threshold_4x4
        self.top_n = top_n
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        self.verbose = verbose
        self.save_tiles = save_tiles
        self._tile_count = 0

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        self._log_path = self.save_dir / f"session_{ts}.csv"
        with open(self._log_path, "w", newline="") as fh:
            csv.writer(fh).writerow(
                ["timestamp", "challenge_type", "target",
                 "tile_idx", "label", "confidence", "clicked"]
            )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def run(
        self,
        page: Page,
        url: str,
        *,
        max_attempts: int = 5,
    ) -> bool:
        """
        Full end-to-end solve loop.  Reloads the page up to *max_attempts*
        times.  Returns True when the reCAPTCHA checkbox is checked.
        """
        from .browser import navigate_and_open  # avoid circular at module level

        for attempt in range(1, max_attempts + 1):
            log.info("Attempt %d / %d", attempt, max_attempts)

            challenge_appeared = navigate_and_open(page, url)
            if not challenge_appeared:
                if self._is_solved(page):
                    log.info("Lucky pass — solved without a visual challenge.")
                    return True
                time.sleep(2.0)
                continue

            # Inner loop: keep solving successive challenge screens
            for _ in range(15):
                outcome = self._solve_one(page)
                if outcome == "solved":
                    log.info("reCAPTCHA solved.")
                    return True
                if outcome == "failed":
                    break  # give up on this page load, try a fresh one

        log.warning("Gave up after %d attempts.", max_attempts)
        return False

    # ------------------------------------------------------------------
    # Internal — challenge routing
    # ------------------------------------------------------------------

    def _solve_one(self, page: Page) -> str:
        """
        Solve one challenge screen.

        Returns
        -------
        "solved"           Checkbox is now checked — we are done.
        "next_challenge"   Submitted OK but Google showed another challenge.
        "failed"           Something went wrong; caller should reload.
        """
        frame = get_challenge_frame(page)
        if frame is None:
            log.warning("Challenge frame not found.")
            return "failed"

        challenge_type = self._detect_type(frame)
        target = self._get_target(frame)

        if target is None:
            log.warning("Unknown challenge target — reloading challenge.")
            try:
                frame.click("#recaptcha-reload-button")
                time.sleep(2.0)
                return "next_challenge"
            except Exception:
                return "failed"

        log.info("Type: %-12s | Target: %s", challenge_type, target)
        if self.verbose:
            print(f"\n[challenge] {challenge_type}  →  target: {target}")

        if challenge_type == "4x4":
            self._solve_grid(page, frame, target, grid_size=4)
        elif challenge_type == "3x3_dynamic":
            self._solve_3x3_dynamic(page, frame, target)
        else:
            self._solve_grid(page, frame, target, grid_size=3)

        time.sleep(random.uniform(0.2, 0.5))
        self._verify(page, frame)
        time.sleep(random.uniform(1.0, 2.0))

        if self._is_solved(page):
            return "solved"

        # Detect empty-submission error ("Please select all matching images").
        # Google keeps the same grid open — reload it so we don't loop on the
        # same blank attempt.
        frame = get_challenge_frame(page)
        if frame:
            try:
                err = frame.query_selector(".rc-imageselect-error-select-more")
                if err:
                    style = err.get_attribute("style") or ""
                    if "none" not in style:
                        log.warning("Empty submission detected — reloading challenge.")
                        frame.click("#recaptcha-reload-button")
                        time.sleep(1.5)
            except Exception:
                pass

        return "next_challenge"

    # ------------------------------------------------------------------
    # Internal — type detection & target parsing
    # ------------------------------------------------------------------

    def _detect_type(self, frame: Frame) -> str:
        try:
            text = frame.inner_text("#rc-imageselect") or ""
        except Exception:
            return "3x3_static"
        lower = text.lower()
        if "squares" in lower:
            return "4x4"
        if "none" in lower:
            return "3x3_dynamic"
        return "3x3_static"

    def _get_target(self, frame: Frame) -> str | None:
        """Read the bold challenge keyword and map it to a model class."""
        try:
            strong = frame.query_selector("#rc-imageselect strong")
            if strong:
                target = _parse_target(strong.inner_text())
                if target:
                    return target
            # Fallback: parse the full description block
            for sel in (
                ".rc-imageselect-desc-no-canonical",
                ".rc-imageselect-desc",
            ):
                el = frame.query_selector(sel)
                if el:
                    target = _parse_target(el.inner_text())
                    if target:
                        return target
        except Exception as exc:
            log.warning("Could not read challenge target: %s", exc)
        return None

    # ------------------------------------------------------------------
    # Internal — grid solvers
    # ------------------------------------------------------------------

    def _solve_grid(
        self,
        page: Page,
        frame: Frame,
        target: str,
        grid_size: int,
    ) -> None:
        """
        One-shot classification for an NxN grid.

        3x3 static: threshold on all_probs[target].
        4x4:        top-N strategy — classify all tiles first, then click the
                    N tiles with the highest target-class probability.  This
                    mirrors what aplesner/Breaking-reCAPTCHAv2 calls
                    USE_TOP_N_STRATEGY and guarantees we always click something
                    rather than submitting an empty grid.
        """
        tiles = self._get_tiles(frame, grid_size * grid_size)
        self._mouse_wander_grid(page, tiles)  # simulate studying the images

        label = f"{grid_size}x{grid_size}"

        if grid_size == 4:
            self._solve_topn(page, tiles, target, label)
        else:
            for i, tile in enumerate(tiles):
                self._process_tile(page, tile, target, label, i)
                time.sleep(random.uniform(0.05, 0.15))

    def _solve_topn(
        self,
        page: Page,
        tiles: list,
        target: str,
        label: str,
    ) -> None:
        """
        Classify all tiles, rank by all_probs[target], click the top-N.

        Minimum floor of 0.05 prevents clicking tiles where the target class
        has essentially zero probability.  N is self.top_n (default 3).
        """
        ranked: list[tuple[float, int, object]] = []  # (target_prob, idx, tile)

        for i, tile in enumerate(tiles):
            result = self._classify_tile(tile, i)
            if result is None:
                continue
            target_prob = result.all_probs.get(target, 0.0)
            top_class = self.predictor.class_names[result.class_idx]
            if self.verbose:
                print(
                    f"    [  scan ] tile {i:2d} → "
                    f"{top_class:<14} conf={result.confidence:.3f}"
                    f"  p({target})={target_prob:.3f}"
                )
            self._write_log(label, target, i, top_class, target_prob, False)
            ranked.append((target_prob, i, tile))
            time.sleep(random.uniform(0.03, 0.08))

        # Sort by target probability descending, take top-N above floor
        ranked.sort(key=lambda x: x[0], reverse=True)
        to_click = [(prob, idx, tile) for prob, idx, tile in ranked
                    if prob >= 0.05][:self.top_n]
        # Click in index order (looks natural — left-to-right, top-to-bottom)
        to_click.sort(key=lambda x: x[1])

        for prob, idx, tile in to_click:
            if self.verbose:
                print(f"    [✓ CLICK] tile {idx:2d}  p({target})={prob:.3f}")
            self._write_log(label, target, idx, target, prob, True)
            self._click_tile(page, tile)
            time.sleep(random.uniform(0.08, 0.20))

    def _solve_3x3_dynamic(
        self,
        page: Page,
        frame: Frame,
        target: str,
    ) -> None:
        """
        Dynamic 3x3: clicked tiles are replaced by new images.
        Re-check each replaced slot until no new matches appear.
        """
        tiles = self._get_tiles(frame, 9)
        self._mouse_wander_grid(page, tiles)
        to_recheck: list[int] = []

        # First pass
        for i, tile in enumerate(tiles):
            clicked = self._process_tile(page, tile, target, "3x3_dynamic", i)
            if clicked:
                to_recheck.append(i)
            time.sleep(random.uniform(0.08, 0.20))

        # Re-check rounds — new images fade in over ~1.5s
        for _ in range(6):
            if not to_recheck:
                break
            time.sleep(random.uniform(1.5, 2.0))
            tiles = self._get_tiles(frame, 9)
            next_round: list[int] = []
            for i in to_recheck:
                if i >= len(tiles):
                    continue
                clicked = self._process_tile(page, tiles[i], target, "3x3_dynamic", i)
                if clicked:
                    next_round.append(i)
                time.sleep(random.uniform(0.08, 0.20))
            to_recheck = next_round

    # ------------------------------------------------------------------
    # Internal — tile helpers
    # ------------------------------------------------------------------

    def _get_tiles(self, frame: Frame, n: int) -> list:
        """
        Return up to *n* tile ElementHandles.

        Selector priority:
          1. td.rc-imageselect-tile  — class present on every tile in all
                                       known reCAPTCHA variants (most reliable)
          2. td[tabindex]            — tabindex 4…4+n-1 (reference approach)
          3. td[tabindex] anywhere   — broadest fallback
        """
        # Wait for at least one tile to be present before querying
        try:
            frame.wait_for_selector("td.rc-imageselect-tile", timeout=5_000)
        except Exception:
            pass

        # 1. Class-based (preferred)
        tiles = frame.query_selector_all("td.rc-imageselect-tile") or []
        if tiles:
            if self.verbose:
                print(f"    [tiles] found {len(tiles)} via class selector")
            return tiles[:n]

        # 2. tabindex-based (matches reference project's XPath approach)
        tiles = []
        for i in range(4, 4 + n):
            el = frame.query_selector(f"td[tabindex='{i}']")
            if el:
                tiles.append(el)
        if tiles:
            if self.verbose:
                print(f"    [tiles] found {len(tiles)} via tabindex selector")
            return tiles[:n]

        # 3. Any td with a tabindex in the challenge frame
        tiles = frame.query_selector_all("td[tabindex]") or []
        if self.verbose:
            print(f"    [tiles] found {len(tiles)} via broad td[tabindex] fallback")
        return tiles[:n]

    def _classify_tile(self, tile, tile_idx: int):
        """
        Screenshot a tile and run the classifier.  Returns a PredictionResult
        or None on failure.  Does NOT click.
        """
        self._tile_count += 1

        if self.save_tiles:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S%f")
            save_path = self.save_dir / f"tile_{ts}_{tile_idx}.png"
        else:
            save_path = self.save_dir / "_tmp_tile.png"

        try:
            tile.screenshot(path=str(save_path))
        except Exception as exc:
            log.warning("Screenshot failed for tile %d: %s", tile_idx, exc)
            return None

        try:
            return self.predictor.predict_image(save_path)
        except Exception as exc:
            log.warning("Prediction failed for tile %d: %s", tile_idx, exc)
            return None

    def _process_tile(
        self,
        page: Page,
        tile,
        target: str,
        challenge_type: str,
        tile_idx: int,
    ) -> bool:
        """
        Screenshot → classify → click if all_probs[target] >= threshold.

        Uses the probability the model assigns to the TARGET class directly
        (same as the reference project) rather than checking whether the
        argmax class equals target.  This catches tiles like:
            top: Car (0.55) | Motorcycle: 0.45  → matched at thr=0.35
        which the old argmax check would incorrectly skip.

        Returns True when the tile was clicked.
        """
        result = self._classify_tile(tile, tile_idx)
        if result is None:
            return False

        target_prob = result.all_probs.get(target, 0.0)
        top_class = self.predictor.class_names[result.class_idx]
        matched = target_prob >= self.predictor.threshold

        if self.verbose:
            tag = "✓ CLICK" if matched else "  skip "
            print(
                f"    [{tag}] tile {tile_idx:2d} → "
                f"{top_class:<14} conf={result.confidence:.3f}"
                f"  p({target})={target_prob:.3f}"
            )

        self._write_log(challenge_type, target, tile_idx, top_class, target_prob, matched)

        if matched:
            self._click_tile(page, tile)
        return matched

    def _mouse_wander_grid(self, page: Page, tiles: list) -> None:
        """
        Move the mouse over a few random tiles before classifying.
        Simulates a human studying the challenge — reduces risk score.
        """
        if not tiles:
            return
        sample = random.sample(tiles, min(3, len(tiles)))
        for tile in sample:
            bbox = tile.bounding_box()
            if bbox:
                cx = bbox["x"] + bbox["width"] / 2 + random.uniform(-8, 8)
                cy = bbox["y"] + bbox["height"] / 2 + random.uniform(-8, 8)
                human_move(page, cx, cy)
                time.sleep(random.uniform(0.08, 0.20))

    def _click_tile(self, page: Page, tile) -> None:
        """Human-like click using the tile's page-level bounding box."""
        bbox = tile.bounding_box()
        if bbox is None:
            tile.click()
            return
        cx = bbox["x"] + bbox["width"] / 2 + random.uniform(-4, 4)
        cy = bbox["y"] + bbox["height"] / 2 + random.uniform(-4, 4)
        human_click(page, cx, cy)

    def _verify(self, page: Page, frame: Frame) -> None:
        """Click the verify button with a human-like move."""
        try:
            btn = frame.query_selector("#recaptcha-verify-button")
            if btn:
                bbox = btn.bounding_box()
                if bbox:
                    cx = bbox["x"] + bbox["width"] / 2
                    cy = bbox["y"] + bbox["height"] / 2
                    human_click(page, cx, cy)
                    return
            frame.click("#recaptcha-verify-button")
        except Exception as exc:
            log.warning("Could not click verify button: %s", exc)

    def _is_solved(self, page: Page) -> bool:
        """Return True when the reCAPTCHA checkbox reports aria-checked=true."""
        try:
            cb_frame = get_checkbox_frame(page)
            if cb_frame:
                anchor = cb_frame.query_selector("#recaptcha-anchor")
                if anchor:
                    return anchor.get_attribute("aria-checked") == "true"
        except Exception:
            pass
        return False

    # ------------------------------------------------------------------
    # Internal — logging
    # ------------------------------------------------------------------

    def _write_log(
        self,
        challenge_type: str,
        target: str,
        tile_idx: int,
        label: str,
        confidence: float,
        clicked: bool,
    ) -> None:
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self._log_path, "a", newline="") as fh:
            csv.writer(fh).writerow(
                [ts, challenge_type, target, tile_idx,
                 label, f"{confidence:.4f}", clicked]
            )
