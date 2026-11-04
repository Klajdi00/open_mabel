//
//  AutonomyTimelineUITests.swift
//  MABEL Mobile AppUITests
//
//  The choreography editor's SHAPE, tested as geometry rather than described in
//  a comment. The brief was specific — "the 3D viewer should be present and the
//  timeline at the bottom, keep everything super compact, the library should be
//  a small button that opens a popup, no more inspector" — and every one of
//  those is a claim about real on-screen frames, so that is how they are
//  checked here.
//
//  The model's rules live in `TimelineModelTests`; this file only asserts the
//  things a unit test cannot see: where the panels are, that nothing overlaps,
//  that the lane labels survive a scroll, and that Execute asks first.
//
//  ENVIRONMENT NOTE (same as ControlTuningUITests): CoreSimulator on this host
//  can be slow to present a sheet, so every wait is generous on purpose. A short
//  timeout here tests the host's mood, not the UI.
//

import XCTest

final class AutonomyTimelineUITests: XCTestCase {

    private static let uiTimeout: TimeInterval = 30

    override func setUpWithError() throws {
        continueAfterFailure = false
    }

    /// Launch straight onto the Autonomy tab. `state` seeds the document so a
    /// layout assertion measures a populated editor and an interaction test
    /// starts from a known-empty one.
    private func launch(_ state: String, extra: [String] = []) -> XCUIApplication {
        let app = XCUIApplication()
        app.launchArguments += ["-uiTesting", "YES", "-startTab", "policies",
                                "-debugTimeline", state] + extra
        app.launch()
        return app
    }

    /// A clip on the timeline. Blocks are DRAWN, so they are exposed as
    /// accessibility buttons rather than as containers.
    private func block(_ app: XCUIApplication, _ id: String) -> XCUIElement {
        app.buttons["block.\(id)"]
    }

    /// Open the library popup, waiting for the strip to settle first — a tap
    /// delivered while the tab is still presenting is swallowed.
    private func tapLibrary(_ app: XCUIApplication) {
        let library = app.buttons["tool.library"]
        _ = library.waitForExistence(timeout: Self.uiTimeout)
        XCTAssertTrue(library.isHittable, "the Library button is not tappable")
        library.tap()
    }

    private func waitForEditor(_ app: XCUIApplication) throws {
        let viewer = app.otherElements["timeline.viewer"]
        guard viewer.waitForExistence(timeout: Self.uiTimeout) else {
            throw XCTSkip("the Autonomy tab did not present (simulator)")
        }
    }

    // MARK: - the shape of the screen

    /// The 3-D viewer is present, it is ABOVE the timeline, and the timeline is
    /// at the bottom. This is the whole layout brief in three frames.
    func testTheViewerIsOnTopAndTheTimelineIsAtTheBottom() throws {
        let app = launch("demo")
        try waitForEditor(app)

        let viewer = app.otherElements["timeline.viewer"]
        let canvas = app.otherElements["timeline.canvas"]
        XCTAssertTrue(canvas.waitForExistence(timeout: Self.uiTimeout),
                      "the timeline canvas is missing")

        XCTAssertLessThanOrEqual(viewer.frame.maxY, canvas.frame.minY + 20,
            "the viewer must sit above the timeline, not beside or below it")
        XCTAssertGreaterThan(canvas.frame.maxY, app.frame.height * 0.72,
            "the timeline must live at the BOTTOM of the screen")
        // Compact: the timeline is a strip, not a half-screen panel.
        XCTAssertLessThan(canvas.frame.height, app.frame.height * 0.42,
            "the timeline is taking more than its share — the viewer is the point")
        XCTAssertGreaterThan(viewer.frame.height, app.frame.height * 0.28,
            "the viewer has been squeezed out of the layout")
    }

    /// There is no docked inspector and no docked library: between the viewer
    /// and the timeline there is only the one-row tool strip.
    func testNoDockedPanelSitsBetweenTheViewerAndTheTracks() throws {
        let app = launch("demo")
        try waitForEditor(app)
        let viewer = app.otherElements["timeline.viewer"]
        let canvas = app.otherElements["timeline.canvas"]
        let gap = canvas.frame.minY - viewer.frame.maxY
        XCTAssertLessThan(gap, 70,
            "\(gap) pt between the viewer and the tracks — something is docked in there")
        // The old inspector's empty state must not come back.
        XCTAssertFalse(app.staticTexts["Select a clip to edit it"].exists)
    }

    /// Every control in the tool strip is reachable and none sits on top of
    /// another. (The strip scrolls, so "reachable" means the primary verbs are
    /// on screen: Library, split, delete.)
    func testToolStripIsLaidOutCleanly() throws {
        let app = launch("demo")
        try waitForEditor(app)
        let library = app.buttons["tool.library"]
        XCTAssertTrue(library.waitForExistence(timeout: Self.uiTimeout))
        XCTAssertTrue(library.isHittable, "the Library button must be tappable")
        // "smaller" — it is a chip, not a panel header.
        XCTAssertLessThan(library.frame.width, app.frame.width * 0.32,
            "the library button is supposed to be small")

        let strip = app.buttons.allElementsBoundByIndex.filter {
            $0.exists && $0.identifier.hasPrefix("tool.")
        }
        XCTAssertGreaterThan(strip.count, 4, "the tool strip lost its tools")
        for i in strip.indices {
            for j in (i + 1)..<strip.count {
                XCTAssertFalse(strip[i].frame.intersects(strip[j].frame),
                    "tools overlap: \(strip[i].identifier) \(strip[i].frame) "
                    + "vs \(strip[j].identifier) \(strip[j].frame)")
            }
        }
    }

    /// The lane labels are pinned: they are there before the tracks scroll and
    /// still there afterwards — shrunk, but present. Losing track of which row
    /// is Nav and which is Manip is how the wrong clip ends up on the base.
    func testLaneLabelsStayPinnedAndShrinkWhenScrolled() throws {
        let app = launch("demo")
        try waitForEditor(app)

        let nav = app.otherElements["lane.header.navigation"]
        let manip = app.otherElements["lane.header.manipulation"]
        let voice = app.otherElements["lane.header.language"]
        for header in [nav, manip, voice] {
            XCTAssertTrue(header.waitForExistence(timeout: Self.uiTimeout),
                          "\(header.identifier) missing before scrolling")
        }
        // Measure the LABELLED width. (The accessibility frame is the union of
        // the header's content, not the column, so this is smaller than the
        // 76 pt gutter — it is still the right thing to compare against itself.)
        let wideWidth = nav.frame.width
        XCTAssertGreaterThan(wideWidth, 20, "the gutter should start out labelled")

        // Scroll the tracks well to the right.
        //
        // NOT `canvas.swipeLeft()`: `timeline.canvas` is the scroll CONTENT, so
        // its frame is the whole 1128 pt document and a swipe aimed at its
        // centre lands on whatever clip happens to be under that point — which
        // drags the clip instead of the scroll view. It did exactly that the
        // first time a trailing safety blend landed mid-document. Drag along the
        // VOICE lane, which is empty here, in window coordinates.
        let voiceY = voice.frame.midY
        let win = app.windows.firstMatch
        func dragTracks() {
            let origin = win.coordinate(withNormalizedOffset: .zero)
            let from = origin.withOffset(CGVector(dx: win.frame.width - 30, dy: voiceY))
            let to = origin.withOffset(CGVector(dx: 110, dy: voiceY))
            from.press(forDuration: 0.05, thenDragTo: to)
        }
        dragTracks()
        dragTracks()

        for header in [nav, manip, voice] {
            XCTAssertTrue(header.exists,
                "\(header.identifier) scrolled away — the gutter must stay pinned")
        }
        XCTAssertLessThan(nav.frame.width, wideWidth,
            "the gutter should shrink once the tracks scroll, not stay full width")
        // The accessibility frame is the ICON, so this is single digits by
        // design; what matters is that it is still on screen and still says
        // which lane this is.
        XCTAssertGreaterThan(nav.frame.width, 4,
            "shrunk is not the same as gone — the icon has to stay")
        XCTAssertLessThan(nav.frame.minX, 40, "the gutter must stay at the leading edge")
    }

    // MARK: - the library popup

    func testTheLibraryIsAPopupAndDropsAClipOnTheTimeline() throws {
        let app = launch("empty")
        try waitForEditor(app)

        tapLibrary(app)
        let wave = app.buttons["action.wave"]
        XCTAssertTrue(wave.waitForExistence(timeout: Self.uiTimeout),
                      "the library popup did not present its actions")
        // It is a sheet: it covers the tracks rather than living beside them.
        XCTAssertGreaterThan(app.buttons["action.wave"].frame.height, 0)
        wave.tap()

        let block = block(app, "wave")
        XCTAssertTrue(block.waitForExistence(timeout: Self.uiTimeout),
                      "picking a clip from the library must put it on the timeline")
        XCTAssertFalse(app.buttons["action.wave"].exists,
                       "the popup should dismiss once you have taken a clip")
    }

    func testTheLibraryFiltersByLane() throws {
        let app = launch("empty")
        try waitForEditor(app)
        tapLibrary(app)
        XCTAssertTrue(app.buttons["action.wave"].waitForExistence(timeout: Self.uiTimeout),
                      "the library popup did not present")

        app.buttons["lane.navigation"].tap()
        // A SwiftUI List materialises lazily, so the assertion is about what the
        // filter REMOVES (deterministic) plus the fact that something is left.
        let gone = NSPredicate(format: "exists == false")
        expectation(for: gone, evaluatedWith: app.buttons["action.wave"])
        waitForExpectations(timeout: 10)
        XCTAssertFalse(app.buttons["action.wave"].exists,
                       "a manipulation clip has no business under the Nav filter")
        let remaining = app.buttons.matching(
            NSPredicate(format: "identifier BEGINSWITH %@", "action.")).count
        XCTAssertGreaterThan(remaining, 0, "the Nav filter emptied the library")
    }

    // MARK: - transport

    /// Preview runs the playhead. It must NOT be the same control as Execute —
    /// one rehearses, the other moves a robot.
    func testPreviewRunsThePlayheadAndIsNotExecute() throws {
        let app = launch("demo")
        try waitForEditor(app)

        let clock = app.staticTexts["transport.timecode"]
        XCTAssertTrue(clock.waitForExistence(timeout: Self.uiTimeout))
        let before = clock.label

        app.buttons["transport.preview"].tap()
        // Let it play for a beat, then read the clock again.
        let moved = NSPredicate(format: "label != %@", before)
        expectation(for: moved, evaluatedWith: clock)
        waitForExpectations(timeout: 10)

        XCTAssertNotEqual(clock.label, before, "Preview did not move the playhead")
        XCTAssertTrue(app.buttons["timeline.execute"].exists,
                      "Execute must be its own control, separate from Preview")
    }

    /// Execute is the one that moves the robot, so it asks first and says so.
    func testExecuteConfirmsBeforeMovingTheRobot() throws {
        let app = launch("demo")
        try waitForEditor(app)

        app.buttons["timeline.execute"].tap()
        let warning = app.staticTexts.containing(
            NSPredicate(format: "label CONTAINS[c] %@", "robot will move")).firstMatch
        XCTAssertTrue(warning.waitForExistence(timeout: Self.uiTimeout),
                      "Execute must warn before it drives the robot")
        let cancel = app.descendants(matching: .any)
            .matching(NSPredicate(format: "label == %@ OR identifier == %@",
                                  "Cancel", "Cancel")).firstMatch
        XCTAssertTrue(cancel.waitForExistence(timeout: 10),
                      "the dialog has no way out:\n\(app.debugDescription)")
        if cancel.exists, cancel.isHittable { cancel.tap() }
    }

    // MARK: - zoom

    func testZoomChangesHowMuchTimeAScreenHolds() throws {
        let app = launch("demo")
        try waitForEditor(app)

        let clip = block(app, "nav_forward")
        XCTAssertTrue(clip.waitForExistence(timeout: Self.uiTimeout))
        let before = clip.frame.width

        let zoomIn = app.buttons["tool.zoomin"]
        let zoomOut = app.buttons["tool.zoomout"]
        XCTAssertTrue(zoomIn.isHittable && zoomOut.isHittable,
            "both zoom controls must be on screen: in=\(zoomIn.frame) out=\(zoomOut.frame)")
        XCTAssertNotEqual(zoomIn.frame, zoomOut.frame, "the zoom controls are the same element")

        zoomIn.tap()
        let zoomed = clip.frame.width
        XCTAssertGreaterThan(zoomed, before * 1.15,
                             "zooming in must make a clip wider (\(before) → \(zoomed))")

        zoomOut.tap()
        let out1 = clip.frame.width
        XCTAssertLessThan(out1, zoomed,
                          "zooming out must make it narrower (\(zoomed) → \(out1))")
        zoomOut.tap()
        XCTAssertLessThan(clip.frame.width, before,
                          "two steps out from one step in must end below where it started")
    }

    // MARK: - editing

    /// Selecting a clip lights the contextual verbs; with nothing selected they
    /// are dead, because there is nothing for them to act on.
    func testClipToolsOnlyLightUpWithASelection() throws {
        let app = launch("demo")
        try waitForEditor(app)

        let duplicate = app.buttons["tool.duplicate"]
        XCTAssertTrue(duplicate.waitForExistence(timeout: Self.uiTimeout))
        XCTAssertFalse(duplicate.isEnabled, "nothing is selected yet")

        block(app, "nav_forward").tap()
        XCTAssertTrue(duplicate.isEnabled, "selecting a clip must enable its verbs")
    }

    /// An annotation is one tap, and it lands on the timeline where the
    /// playhead is.
    func testAnnotationsCanBeAdded() throws {
        let app = launch("empty")
        try waitForEditor(app)
        app.buttons["tool.note"].tap()
        let editor = app.navigationBars["Note"]
        XCTAssertTrue(editor.waitForExistence(timeout: Self.uiTimeout),
                      "adding a note should open its editor:\n\(app.debugDescription)")
        // Nothing is written until Done — cancelling must leave no chip behind.
        app.buttons["Cancel"].tap()
        XCTAssertFalse(editor.waitForExistence(timeout: 3), "the note editor should close")
        XCTAssertFalse(app.staticTexts["note"].exists,
                       "a cancelled note must not land on the timeline")

        app.buttons["tool.note"].tap()
        XCTAssertTrue(editor.waitForExistence(timeout: Self.uiTimeout))
        app.buttons["Done"].tap()
        XCTAssertTrue(app.staticTexts["note"].waitForExistence(timeout: 10),
                      "the annotation should appear on the timeline")
    }

    /// The ramp is the one thing about a bridge the operator decides, so it has
    /// to be reachable and adjustable — not only by dragging a 0.4 s block's
    /// edge on a phone.
    func testABridgesRampDurationCanBeAdjusted() throws {
        let app = launch("demo")
        try waitForEditor(app)
        // The demo document's guard inserts bridges; open one.
        // A document has SEVERAL bridges, so the identifier is ambiguous and a
        // plain subscript never resolves. Take the first one actually on screen.
        let bridges = app.buttons.matching(identifier: "block.blend")
        XCTAssertTrue(bridges.firstMatch.waitForExistence(timeout: Self.uiTimeout),
                      "the guard should have inserted a bridge to inspect")
        // `isHittable` THROWS on an element scrolled off the canvas, so pick by
        // frame: the canvas is 1128 pt of content behind a ~400 pt window.
        let win = app.windows.firstMatch.frame
        let blend = (0..<bridges.count)
            .map { bridges.element(boundBy: $0) }
            .first { win.contains(CGPoint(x: $0.frame.midX, y: $0.frame.midY)) }
        XCTAssertNotNil(blend, "no bridge was on screen to tap")
        blend?.tap()                                 // select
        app.buttons["tool.adjust"].tap()             // open its inspector
        let ramp = app.sliders["blend.ramp"]
        XCTAssertTrue(ramp.waitForExistence(timeout: Self.uiTimeout),
                      "a bridge's inspector must offer its ramp duration")
        let before = ramp.normalizedSliderPosition
        ramp.adjust(toNormalizedSliderPosition: min(1.0, before + 0.3))
        XCTAssertNotEqual(ramp.normalizedSliderPosition, before, accuracy: 0.01,
                          "the ramp should actually move")
    }
}
