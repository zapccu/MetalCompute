//
//  MetalConvertible.swift
//
//  Part of MetalCompute
//
//  Conversion between Swift types and their Metal layout
//

import Metal

/// Common base of MetalEncodable and MetalDecodable: a Swift type with an associated
/// Metal layout type. Don't conform to it directly.
public protocol MetalLayoutProviding {
    /// Plain struct (numbers, SIMD vectors) with the layout of the Metal struct
    associatedtype MetalLayout

    /// Stride of MetalLayout. Has a default implementation.
    static var metalLayoutStride: Int { get }
}

extension MetalLayoutProviding {

    public static var metalLayoutStride: Int {
        MemoryLayout<MetalLayout>.stride
    }

    /// Stops with a precondition failure if MetalLayout can't be copied 1:1 to Metal
    static func checkMetalLayout() {
        requirePlainType(MetalLayout.self)
    }
}

/// A Swift type which is converted into a Metal layout type before it is passed to a kernel.
///
/// Use it for structs whose Swift layout differs from the Metal struct, e.g. because they
/// contain `Int` or `Double` fields:
///
/// ```swift
/// struct Parameters {                      // Swift side
///     var maxIter: Int
///     var bailout: Double
/// }
///
/// struct MetalParameters {                 // same layout as the Metal struct
///     var maxIter: Int32
///     var bailout: SIMD2<Float>            // f64 (Metal64 Float2)
/// }
///
/// extension Parameters: MetalEncodable {
///     func toMetal() -> MetalParameters {
///         MetalParameters(maxIter: Int32(maxIter), bailout: bailout.toMetal())
///     }
/// }
/// ```
///
/// Single values, arrays and buffer pointers of the type are converted automatically.
/// `MetalLayout` must be a plain struct (numbers, SIMD vectors) with the Metal layout.
public protocol MetalEncodable: MetalArgument, MetalLayoutProviding {
    /// Converts the value into its Metal layout
    func toMetal() -> MetalLayout

    /// Converts `count` values at `elements` (memory bound to Self) and copies them into a new
    /// Metal buffer. Has a default implementation, used for arrays and buffer pointers.
    static func makeMetalBuffer(from elements: UnsafeRawPointer, count: Int,
                                device: any MTLDevice) throws -> any MTLBuffer

    /// Writes `count` copies of the converted value to `destination` (memory with MetalLayout).
    /// Has a default implementation, used for initial values of output buffers.
    func fillMetal(_ destination: UnsafeMutableRawPointer, count: Int)
}

/// A Swift type which is created from a Metal result.
///
/// When the output type of `ComputeKernel.run` conforms to `MetalDecodable`, the output buffer
/// is created with `MetalLayout` elements and every element is converted with `init(fromMetal:)`
/// before the result is returned:
///
/// ```swift
/// struct Result: MetalDecodable {
///     var iterations: Int
///     var distance: Double
///     init(fromMetal value: MetalResult) {
///         iterations = Int(value.iterations)
///         distance = Double(fromMetal: value.distance)
///     }
/// }
///
/// let result: [Result] = try await kernel.run(points, parameters, count: points.count)
/// ```
public protocol MetalDecodable: MetalLayoutProviding {
    /// Creates the value from its Metal layout
    init(fromMetal value: MetalLayout)

    /// Converts `count` values at `source` (memory with MetalLayout) and initializes
    /// `destination` (uninitialized memory for Self). Has a default implementation.
    static func decodeMetal(from source: UnsafeRawPointer, into destination: UnsafeMutableRawPointer, count: Int)
}

/// A type which can be converted in both directions
public typealias MetalConvertible = MetalEncodable & MetalDecodable

// MARK: - Default implementations

extension MetalEncodable {

    /// Converts the value with toMetal() and passes the Metal layout
    public func bind(to encoder: any MTLComputeCommandEncoder, index: Int, device: any MTLDevice) throws {
        try bindValue(toMetal(), to: encoder, index: index, device: device)
    }

    /// Stride of the Metal layout
    public var metalStride: Int {
        MemoryLayout<MetalLayout>.stride
    }

    public static func makeMetalBuffer(from elements: UnsafeRawPointer, count: Int,
                                       device: any MTLDevice) throws -> any MTLBuffer {
        requirePlainType(MetalLayout.self)
        let values = UnsafeBufferPointer(start: elements.assumingMemoryBound(to: Self.self), count: count)
        let converted = values.map { $0.toMetal() }
        return try converted.withUnsafeBytes { try makeBuffer($0, device: device) }
    }

    public func fillMetal(_ destination: UnsafeMutableRawPointer, count: Int) {
        let value = toMetal()
        destination.bindMemory(to: MetalLayout.self, capacity: count).update(repeating: value, count: count)
    }
}

extension MetalDecodable {

    public static func decodeMetal(from source: UnsafeRawPointer, into destination: UnsafeMutableRawPointer, count: Int) {
        let values = source.bindMemory(to: MetalLayout.self, capacity: count)
        let results = destination.assumingMemoryBound(to: Self.self)
        for index in 0..<count {
            (results + index).initialize(to: Self(fromMetal: values[index]))
        }
    }
}

// MARK: - Output buffer

/// Output buffer of a kernel run. Handles plain output types (copied 1:1) and
/// MetalDecodable output types (buffer with MetalLayout, converted after the run).
struct OutputBuffer<Output> {

    let buffer: any MTLBuffer
    let count: Int

    /// Stride of one output element in Metal
    let metalStride: Int

    private let decodable: (any MetalDecodable.Type)?

    init(device: any MTLDevice, count: Int, initialValue: Output?) throws {
        guard count > 0 else { throw MetalComputeError.invalidCount(count) }

        decodable = Output.self as? any MetalDecodable.Type
        if let decodable {
            decodable.checkMetalLayout()
            metalStride = decodable.metalLayoutStride
        }
        else {
            requirePlainType(Output.self)
            metalStride = MemoryLayout<Output>.stride
        }

        let length = metalStride * count
        guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
            throw MetalComputeError.bufferCreationFailed(length: length)
        }
        self.buffer = buffer
        self.count = count

        if let initialValue {
            if decodable != nil {
                guard let encodable = initialValue as? any MetalEncodable else {
                    preconditionFailure("initialValue: \(Output.self) must conform to MetalEncodable to be converted")
                }
                encodable.fillMetal(buffer.contents(), count: count)
            }
            else {
                buffer.contents().bindMemory(to: Output.self, capacity: count).update(repeating: initialValue, count: count)
            }
        }
    }

    /// The output as Swift array, MetalDecodable elements are converted
    func result() -> [Output] {
        if let decodable {
            let source = UnsafeRawPointer(buffer.contents())
            return [Output](unsafeUninitializedCapacity: count) { destination, initializedCount in
                decodable.decodeMetal(from: source, into: UnsafeMutableRawPointer(destination.baseAddress!), count: count)
                initializedCount = count
            }
        }
        let values = buffer.contents().bindMemory(to: Output.self, capacity: count)
        return Array(UnsafeBufferPointer(start: values, count: count))
    }
}
