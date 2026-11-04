//
//  PLYMeshTests.swift
//  Every rig PLY the app ships must decode completely. The NovaRig exports put
//  colour on the FACE element (count + indices + r,g,b,a); the parser used to
//  assume a `1 + n*4` face record, read garbage from face 2 on, and a negative
//  index trapped `UInt32(_:)` — the app crashed on launch whenever it loaded
//  NOVA (2026-10-14).
//

import XCTest
@testable import MabelTeleop

final class PLYMeshTests: XCTestCase {

    private func header(_ url: URL) -> (vertices: Int, faces: Int)? {
        guard let d = try? Data(contentsOf: url),
              let r = d.range(of: Data("end_header\n".utf8)),
              let h = String(data: d[..<r.lowerBound], encoding: .ascii) else { return nil }
        var v = 0, f = 0
        for line in h.split(separator: "\n") {
            let p = line.split(separator: " ")
            if p.count >= 3, p[0] == "element" {
                if p[1] == "vertex" { v = Int(p[2]) ?? 0 }
                if p[1] == "face" { f = Int(p[2]) ?? 0 }
            }
        }
        return (v, f)
    }

    func test_everyBundledRigPLY_decodesEveryFace() throws {
        let urls = Bundle.main.urls(forResourcesWithExtension: "ply", subdirectory: nil) ?? []
        XCTAssertGreaterThan(urls.count, 100, "rig PLYs missing from the bundle")
        var nova = 0
        for url in urls {
            let h = try XCTUnwrap(header(url), url.lastPathComponent)
            let g = try XCTUnwrap(PLYMesh.parse(url: url), url.lastPathComponent)
            XCTAssertEqual(g.positions.count, h.vertices, url.lastPathComponent)
            // trimesh exports are all triangles: one triangle per face, none dropped.
            XCTAssertEqual(g.indices.count, 3 * h.faces, url.lastPathComponent)
            XCTAssertTrue(g.indices.allSatisfy { Int($0) < h.vertices }, url.lastPathComponent)
            if url.lastPathComponent.hasPrefix("nova_") { nova += 1 }
        }
        XCTAssertGreaterThan(nova, 0, "NovaRig PLYs (face-coloured) not exercised")
    }

    /// A corrupt asset costs that face, never the app.
    func test_outOfRangeIndex_dropsTheFace_doesNotTrap() throws {
        var d = Data("""
        ply
        format binary_little_endian 1.0
        element vertex 3
        property float x
        property float y
        property float z
        element face 2
        property list uchar int vertex_indices
        end_header

        """.utf8)
        for f: Float in [0, 0, 0, 1, 0, 0, 0, 1, 0] { withUnsafeBytes(of: f) { d.append(contentsOf: $0) } }
        for face: [Int32] in [[0, 1, 2], [0, -7, 99]] {
            d.append(3)
            for i in face { withUnsafeBytes(of: i) { d.append(contentsOf: $0) } }
        }
        let url = FileManager.default.temporaryDirectory.appendingPathComponent("bad.ply")
        try d.write(to: url)
        let g = try XCTUnwrap(PLYMesh.parse(url: url))
        XCTAssertEqual(g.indices, [0, 1, 2])
    }
}
