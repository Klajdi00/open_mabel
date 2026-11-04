//
//  EpisodeNamingTests.swift
//  The one-take recording flow (2026-10-14): clip names, folder sanitising and
//  "Episode N" labels must agree with the bridge (sim_teleop_bridge.py) and
//  CurationRecorder._safe. Same text in both apps bar the import line.
//

import XCTest
@testable import MABEL_Mobile_App

final class EpisodeNamingTests: XCTestCase {

    private func clip(_ n: String) -> RobotState.DemoClipInfo {
        .init(name: n, duration: 2, frames: 40, created: "", recordedBy: "", joints: 4)
    }

    /// The preview must count the folder the SERVER will use (CurationRecorder._safe).
    func test_safeFolder_mirrorsTheServer() {
        XCTAssertEqual(EpisodeNaming.safeFolder("  pick cube "), "pick_cube")
        XCTAssertEqual(EpisodeNaming.safeFolder("pick/cube!"), "pickcube")
        XCTAssertEqual(EpisodeNaming.safeFolder("a-b_c"), "a-b_c")
        XCTAssertEqual(EpisodeNaming.safeFolder("   "), "")
    }

    func test_parse_roundTripsTheBridgesClipName() {
        let p = EpisodeNaming.parse("pick_cube_episode_0003")
        XCTAssertEqual(p?.folder, "pick_cube"); XCTAssertEqual(p?.index, 3)
        // A folder may itself contain "_episode_" — the LAST one is the suffix.
        XCTAssertEqual(EpisodeNaming.parse("x_episode_y_episode_0001")?.folder, "x_episode_y")
        XCTAssertNil(EpisodeNaming.parse("demo_1423"))
        XCTAssertNil(EpisodeNaming.parse("_episode_0001"))
    }

    /// Files are 0-based; people count from 1. episode_0000.h5 is "Episode 1".
    func test_label_isOneBased() {
        XCTAssertEqual(EpisodeNaming.label(0), "Episode 1")
        XCTAssertEqual(EpisodeNaming.label(99), "Episode 100")
    }

    func test_groups_byFolder_newestFirst_otherLast() {
        let g = EpisodeNaming.groups([clip("b_episode_0000"), clip("demo_1423"),
                                      clip("a_episode_0000"), clip("b_episode_0002"),
                                      clip("b_episode_0001")])
        XCTAssertEqual(g.map(\.folder), ["a", "b", EpisodeNaming.otherGroup])
        XCTAssertEqual(g[1].clips.map(\.name),
                       ["b_episode_0002", "b_episode_0001", "b_episode_0000"])
    }
}

