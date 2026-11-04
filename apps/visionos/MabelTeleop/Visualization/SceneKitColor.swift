//
//  SceneKitColor.swift
//  MABEL Mobile App
//
//  Shared UIColor hex helper for the SceneKit visualizations.
//

import UIKit

extension UIColor {
    /// Construct from a 0xRRGGBB literal.
    convenience init(rgb: UInt32) {
        self.init(
            red:   CGFloat((rgb >> 16) & 0xFF) / 255,
            green: CGFloat((rgb >> 8) & 0xFF) / 255,
            blue:  CGFloat(rgb & 0xFF) / 255,
            alpha: 1
        )
    }
}
