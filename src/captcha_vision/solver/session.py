"""
Core reCAPTCHA solving logic.

Challenge taxonomy (same three types handled by aplesner/Breaking-reCAPTCHAv2):
  3x3_static   — Select all images matching X (9 tiles, one-shot).
  3x3_dynamic  — Select all images matching X; clicked tiles refresh with new
                 images (identified by "none" in the wrapper text).
  4x4          — 4×4 grid of tiles from a single large image ("squares" in
                 wrapper text).  Multi-scale classification: full composite,
                 overlapping 2×2 blocks, and individual tiles are combined.
"""
from __future__ import annotations

import csv
import logging
import random
import time
from io import BytesIO
from datetime import datetime
from pathlib import Path
from urllib.request import Request, urlopen

from PIL import Image
from playwright.sync_api import Frame, Page

from captcha_vision.inference.predictor import Predictor

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
        4x4:        Multi-scale classification — screenshot the full grid,
                    classify at three scales (full, 2×2 blocks, tile), and
                    combine scores for robust per-tile decisions.
        """
        tiles = self._get_tiles(frame, grid_size * grid_size)
        self._mouse_wander_grid(page, tiles)

        if grid_size == 4:
            self._solve_4x4_multiscale(page, frame, tiles, target)
        else:
            label = f"{grid_size}x{grid_size}"
            for i, tile in enumerate(tiles):
                self._process_tile(page, tile, target, label, i)
                time.sleep(random.uniform(0.05, 0.15))

    # ------------------------------------------------------------------
    # Internal — 4×4 multi-scale solver
    # ------------------------------------------------------------------

    _GRID_IMAGE_SELECTORS = [
        "#rc-imageselect-target img",
        "table.rc-imageselect-table-44 img",
        "table.rc-imageselect-table img",
    ]

    _GRID_TABLE_SELECTORS = [
        "table.rc-imageselect-table-44",
        "table.rc-imageselect-table",
        "#rc-imageselect-target",
    ]

    def _load_grid_image(self, frame: Frame) -> Image.Image | None:
        """
        Load the 4×4 challenge image with the cleanest source available.

        Prefer the original ``img`` URL when present so inference sees the raw
        challenge image rather than a browser-rendered screenshot with borders
        and interpolation artefacts. Fall back to screenshots if the source
        image is not directly retrievable.
        """
        for sel in self._GRID_IMAGE_SELECTORS:
            el = frame.query_selector(sel)
            if el is None:
                continue
            try:
                src = el.get_attribute("src")
                if not src:
                    continue
                req = Request(
                    src,
                    headers={
                        "User-Agent": (
                            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                            "AppleWebKit/537.36 (KHTML, like Gecko) "
                            "Chrome/120.0.0.0 Safari/537.36"
                        )
                    },
                )
                with urlopen(req, timeout=10) as resp:
                    return Image.open(BytesIO(resp.read())).convert("RGB")
            except Exception as exc:
                log.debug("Grid image URL fetch failed for %s: %s", sel, exc)
        return self._screenshot_grid(frame)

    def _screenshot_grid(self, frame: Frame) -> Image.Image | None:
        """Screenshot the 4×4 challenge grid table and return a PIL Image."""
        for sel in self._GRID_TABLE_SELECTORS:
            el = frame.query_selector(sel)
            if el is None:
                continue
            try:
                path = self.save_dir / "_tmp_grid.png"
                el.screenshot(path=str(path))
                return Image.open(path).convert("RGB")
            except Exception as exc:
                log.debug("Selector %s screenshot failed: %s", sel, exc)
        return None

    def _solve_4x4_multiscale(
        self,
        page: Page,
        frame: Frame,
        tiles: list,
        target: str,
    ) -> None:
        """
        Multi-scale classification for 4×4 grids.

        Individual 4×4 tiles are ~1/16th of a scene — far too small for an
        image classifier trained on whole tiles.  This method classifies at
        three scales and combines the signals:

        Scale 1 — Full composite (1 classification)
            Confirms the target class is present.  Provides a global
            confidence that adjusts how aggressively tiles are selected.

        Scale 2 — Nine overlapping 2×2 blocks (9 classifications)
            Each block shows 1/4 of the scene — enough to recognise most
            objects.  Localises the target to groups of 4 tiles.

        Scale 3 — Individual tiles (16 classifications)
            Fine-grained tiebreaker.  Within a high-scoring 2×2 block, the
            tile scores distinguish which tiles actually overlap the object.

        The per-tile score is a weighted combination:
            0.55 × max(2×2 blocks containing tile)
          + 0.30 × tile score
          + 0.15 × global score

        Scores then feed into ``_adaptive_select`` (gap-based cutoff +
        spatial coherence).  Falls back to tile-only classification if the
        grid screenshot fails.
        """
        composite = self._load_grid_image(frame)
        if composite is None:
            log.warning("Grid screenshot failed — falling back to tile-only.")
            self._solve_4x4_tile_only(page, tiles, target)
            return

        cw, ch = composite.size
        tw, th = cw / 4.0, ch / 4.0

        # --- Scale 1: full composite ---
        full_result = self.predictor.predict_pil(composite, label="composite")
        global_conf = self.predictor.score_target_pil(composite, target)
        if self.verbose:
            print(
                f"    [composite] p({target})={global_conf:.3f}"
                f"  top={full_result.label} ({full_result.confidence:.3f})"
            )

        # --- Scale 2: nine overlapping 2×2 blocks ---
        block_scores: dict[tuple[int, int], float] = {}
        for br in range(3):
            for bc in range(3):
                crop = composite.crop((
                    int(bc * tw), int(br * th),
                    int((bc + 2) * tw), int((br + 2) * th),
                ))
                result = self.predictor.predict_pil(crop, label=f"block_{br}_{bc}")
                block_scores[(br, bc)] = self.predictor.score_target_pil(crop, target)
                if self.verbose:
                    print(
                        f"    [block {br},{bc}] p({target})="
                        f"{block_scores[(br, bc)]:.3f}"
                        f"  top={result.label} ({result.confidence:.3f})"
                    )

        # --- Scale 3: individual tiles from composite ---
        tile_scores: list[float] = []
        for i in range(16):
            r, c = divmod(i, 4)
            crop = composite.crop((
                int(c * tw), int(r * th),
                int((c + 1) * tw), int((r + 1) * th),
            ))
            result = self.predictor.predict_pil(crop, label=f"tile_{i}")
            tp = self.predictor.score_target_pil(crop, target)
            tile_scores.append(tp)
            if self.verbose:
                top_class = self.predictor.class_names[result.class_idx]
                print(
                    f"    [  tile ] {i:2d}  {top_class:<14}"
                    f"  conf={result.confidence:.3f}  p({target})={tp:.3f}"
                )

        # --- Combine the three scales ---
        scored: list[tuple[float, int, object]] = []
        for i in range(min(16, len(tiles))):
            r, c = divmod(i, 4)

            # Max of all 2×2 blocks that contain this tile.
            # Tile (r, c) is covered by block (br, bc) when
            # br <= r <= br+1 and bc <= c <= bc+1.
            block_max = 0.0
            for br in range(max(0, r - 1), min(3, r + 1)):
                for bc in range(max(0, c - 1), min(3, c + 1)):
                    block_max = max(block_max, block_scores.get((br, bc), 0.0))

            tile_prob = tile_scores[i] if i < len(tile_scores) else 0.0
            combined = 0.55 * block_max + 0.30 * tile_prob + 0.15 * global_conf

            if self.verbose:
                print(
                    f"    [score ] {i:2d}  block_max={block_max:.3f}"
                    f"  tile={tile_prob:.3f}  global={global_conf:.3f}"
                    f"  → {combined:.3f}"
                )

            self._write_log("4x4", target, i, target, combined, False)
            scored.append((combined, i, tiles[i]))

        # --- Select and click ---
        to_click = self._adaptive_select(scored, 4, global_conf)
        to_click.sort(key=lambda x: x[1])

        for prob, idx, tile in to_click:
            if self.verbose:
                print(f"    [✓ CLICK] tile {idx:2d}  score={prob:.3f}")
            self._write_log("4x4", target, idx, target, prob, True)
            self._click_tile(page, tile)
            time.sleep(random.uniform(0.08, 0.20))

    def _solve_4x4_tile_only(
        self,
        page: Page,
        tiles: list,
        target: str,
    ) -> None:
        """Fallback: classify individual tiles when the grid screenshot fails."""
        scored: list[tuple[float, int, object]] = []
        for i, tile in enumerate(tiles):
            result, screenshot_path = self._classify_tile(tile, i)
            if result is None:
                scored.append((0.0, i, tile))
                continue
            if screenshot_path is None:
                target_prob = 0.0
            else:
                target_prob = self.predictor.score_target_image(screenshot_path, target)
            top_class = self.predictor.class_names[result.class_idx]
            if self.verbose:
                print(
                    f"    [  scan ] tile {i:2d} → "
                    f"{top_class:<14} conf={result.confidence:.3f}"
                    f"  p({target})={target_prob:.3f}"
                )
            self._write_log("4x4", target, i, top_class, target_prob, False)
            scored.append((target_prob, i, tile))
            time.sleep(random.uniform(0.03, 0.08))

        to_click = self._adaptive_select(scored, 4, 0.5)
        to_click.sort(key=lambda x: x[1])

        for prob, idx, tile in to_click:
            if self.verbose:
                print(f"    [✓ CLICK] tile {idx:2d}  p({target})={prob:.3f}")
            self._write_log("4x4", target, idx, target, prob, True)
            self._click_tile(page, tile)
            time.sleep(random.uniform(0.08, 0.20))

    # ------------------------------------------------------------------
    # Internal — adaptive 4x4 tile selection
    # ------------------------------------------------------------------

    @staticmethod
    def _neighbours_4x4(idx: int, grid: int = 4) -> list[int]:
        """Return indices of 4-connected neighbours in a grid×grid layout."""
        r, c = divmod(idx, grid)
        nbrs = []
        if r > 0:
            nbrs.append((r - 1) * grid + c)
        if r < grid - 1:
            nbrs.append((r + 1) * grid + c)
        if c > 0:
            nbrs.append(r * grid + c - 1)
        if c < grid - 1:
            nbrs.append(r * grid + c + 1)
        return nbrs

    def _adaptive_select(
        self,
        scored: list[tuple[float, int, object]],
        grid_size: int,
        global_conf: float = 0.5,
    ) -> list[tuple[float, int, object]]:
        """
        Pick which 4x4 tiles to click using score ranking + spatial coherence.

        Works with both raw probabilities (tile-only fallback) and the
        weighted combined scores from multi-scale classification.

        1. Sort by descending score.
        2. Find the largest gap — the natural boundary between target and
           background tiles.  Multi-scale scores produce much cleaner gaps
           than raw per-tile probabilities.
        3. Boost tiles that have high-scoring neighbours (spatial coherence).
        4. Clamp the final count to [2, 8].
        """
        if not scored:
            return []

        by_score = sorted(scored, key=lambda x: x[0], reverse=True)
        scores = [s for s, _, _ in by_score]
        score_map = {idx: s for s, idx, _ in scored}

        # --- Gap-based adaptive cutoff ---
        best_gap = 0.0
        cut_pos = self.top_n
        limit = min(len(scores) - 1, 10)
        for i in range(1, limit):
            gap = scores[i - 1] - scores[i]
            if gap > best_gap and scores[i - 1] >= 0.08:
                best_gap = gap
                cut_pos = i

        # Require a meaningful gap — if the distribution is flat (no clear
        # separation), fall back to top_n.
        if best_gap < 0.04:
            cut_pos = self.top_n

        candidates = set(by_score[i][1] for i in range(cut_pos))

        # --- Spatial coherence boost ---
        for score, idx, tile in by_score[cut_pos:]:
            if score < 0.06:
                break
            nbrs_in = sum(
                1 for n in self._neighbours_4x4(idx, grid_size)
                if n in candidates
            )
            if nbrs_in >= 2:
                candidates.add(idx)

        # --- Remove isolated tiles ---
        # Adaptive threshold: when global confidence is high the target is
        # prominent, so isolated tiles are more plausible.
        if global_conf >= 0.6:
            isolated_thr = 0.30
        elif global_conf <= 0.2:
            isolated_thr = 0.55
        else:
            isolated_thr = 0.40

        final = set()
        for idx in candidates:
            nbrs_in = sum(
                1 for n in self._neighbours_4x4(idx, grid_size)
                if n in candidates
            )
            if nbrs_in > 0 or score_map[idx] >= isolated_thr:
                final.add(idx)

        if len(final) < 2:
            final = set(by_score[i][1] for i in range(min(self.top_n, len(by_score))))

        max_clicks = min(self.top_n + 5, 8)
        if len(final) > max_clicks:
            ranked = sorted(final, key=lambda i: score_map[i], reverse=True)
            final = set(ranked[:max_clicks])

        return [(s, idx, tile) for s, idx, tile in scored if idx in final]

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

    def _classify_tile(self, tile, tile_idx: int, *, use_tta: bool = False):
        """
        Screenshot a tile and run the classifier.  Returns a PredictionResult
        plus the screenshot path, or ``(None, None)`` on failure.  Does NOT click.

        When ``use_tta`` is True (recommended for 4x4 tiles), five-crop +
        flip TTA is applied for more robust partial-object classification.
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
            return None, None

        try:
            if use_tta:
                return self.predictor.predict_image_with_tta(save_path), save_path
            return self.predictor.predict_image(save_path), save_path
        except Exception as exc:
            log.warning("Prediction failed for tile %d: %s", tile_idx, exc)
            return None, None

    def _process_tile(
        self,
        page: Page,
        tile,
        target: str,
        challenge_type: str,
        tile_idx: int,
    ) -> bool:
        """
        Screenshot → classify → click if the target-presence score clears the
        threshold.

        The top-class prediction is still useful for logging, but mixed tiles
        are scored with a target-aware max-over-crops pass so the presence of a
        distractor object does not suppress the target as aggressively.

        Returns True when the tile was clicked.
        """
        result, screenshot_path = self._classify_tile(tile, tile_idx)
        if result is None:
            return False

        if screenshot_path is None:
            target_prob = 0.0
        else:
            target_prob = self.predictor.score_target_image(screenshot_path, target)
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
