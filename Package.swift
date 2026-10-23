// swift-tools-version: 6.0

import PackageDescription

let package = Package(
    name: "MetalCompute",
    platforms: [
        .macOS(.v13),
        .iOS(.v16)
    ],
    products: [
        .library(
            name: "MetalCompute",
            targets: ["MetalCompute"]
        ),
    ],
    targets: [
        .target(
            name: "MetalCompute"
        ),
        .testTarget(
            name: "MetalComputeTests",
            dependencies: ["MetalCompute"]
        ),
    ]
)
