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
    dependencies: [
        // Complex<Double> support (same package as used by Metal64)
        .package(url: "https://github.com/apple/swift-numerics.git", from: "1.0.0")
    ],
    targets: [
        .target(
            name: "MetalCompute",
            dependencies: [
                .product(name: "RealModule", package: "swift-numerics"),
                .product(name: "ComplexModule", package: "swift-numerics")
            ]
        ),
        .testTarget(
            name: "MetalComputeTests",
            dependencies: [
                "MetalCompute",
                .product(name: "ComplexModule", package: "swift-numerics")
            ]
        ),
    ]
)
