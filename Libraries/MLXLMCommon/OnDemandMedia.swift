// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN

/// Where a model reads the media encoder weights it loads on demand.
public struct MediaWeightSource: Sendable {
    public let modelDirectory: URL
    public let weightFileSelection: WeightFileSelection
    public let quantization: BaseConfiguration.Quantization?
    public let perLayerQuantization: BaseConfiguration.PerLayerQuantization?

    public init(
        modelDirectory: URL,
        weightFileSelection: WeightFileSelection = .automatic,
        quantization: BaseConfiguration.Quantization? = nil,
        perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
    ) {
        self.modelDirectory = modelDirectory
        self.weightFileSelection = weightFileSelection
        self.quantization = quantization
        self.perLayerQuantization = perLayerQuantization
    }
}

/// A model whose media encoders, such as a vision or audio tower, load on first use.
///
/// Model factories detach the encoders before loading, so a text-only session never reads
/// their weights. The first input with media calls ``loadMediaEncoders()``, which reads,
/// verifies and evaluates the encoders apart from the model and only then attaches them. The
/// model never holds unevaluated arrays, and loading never touches the language model.
///
/// `sanitize(weights:metadata:)` must not depend on whether the encoders are attached, and
/// ``LanguageModel/prepare()`` does not run again after they load.
public protocol OnDemandMediaEncoders: BaseLanguageModel {
    /// Top-level module keys of the encoders, such as `vision_tower`.
    ///
    /// Checkpoint names of encoder weights contain the key, before and after `sanitize`.
    var mediaEncoderKeys: [String] { get }

    /// New, unloaded encoders keyed by ``mediaEncoderKeys``.
    func makeMediaEncoders() -> [String: Module]

    /// Where ``loadMediaEncoders()`` reads the weights, or `nil` if it cannot load them.
    var mediaWeightSource: MediaWeightSource? { get set }
}

extension OnDemandMediaEncoders {

    /// Whether the encoders are in the module tree.
    public var mediaEncodersAreAttached: Bool {
        mediaEncoderKeys.allSatisfy { key in
            if case .value(.module)? = items()[key] { return true }
            return false
        }
    }

    /// Remove the encoders from the module tree and release their weights.
    ///
    /// The next input with media loads them again from ``mediaWeightSource``.
    public func detachMediaEncoders() throws {
        let detached = mediaEncoderKeys.map { ($0, NestedItem<String, Module>.none) }
        try update(
            modules: ModuleChildren(values: Dictionary(uniqueKeysWithValues: detached)),
            verify: .all)
    }

    /// Remove the encoders so that the first input with media loads them from `source`.
    public func detachMediaEncoders(loadingFrom source: MediaWeightSource) throws {
        mediaWeightSource = source
        try detachMediaEncoders()
    }

    /// Load and attach the encoders if they are detached and ``mediaWeightSource`` is set.
    ///
    /// Call it with exclusive access to the model, as during prefill.
    public func loadMediaEncoders() throws {
        guard !mediaEncodersAreAttached, let source = mediaWeightSource else { return }

        let prefixes = mediaWeightPrefixes
        let urls = try safetensorWeightURLs(
            in: source.modelDirectory,
            selection: source.weightFileSelection,
            additionalFiles: (self as? any AdditionalWeightFilesProviding)?.additionalWeightFiles
                ?? [])
        let (loaded, metadata) = try loadWeightArrays(urls: urls) {
            matchesWeightPrefixes($0, prefixes: prefixes)
        }
        let weights = sanitize(weights: loaded, metadata: metadata)

        let staged = MediaEncoderStage(makeMediaEncoders())
        quantizeCheckpointLayers(
            of: staged, weights: weights, quantization: source.quantization,
            perLayerQuantization: source.perLayerQuantization)
        try staged.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        eval(staged)

        try attach(staged.encoders)
    }

    package func attach(_ encoders: [String: Module]) throws {
        try update(modules: ModuleChildren(values: encoders.mapValues { .value($0) }), verify: .all)
    }

    var mediaWeightPrefixes: [String] { mediaEncoderKeys.map { $0 + "." } }
}

/// Whether `key` names a weight under one of `prefixes`, at its start or after a `.`.
///
/// The second form matches raw checkpoint names with an extra scope, such as
/// `model.vision_tower...`, before `sanitize` removes it.
func matchesWeightPrefixes(_ key: String, prefixes: [String]) -> Bool {
    prefixes.contains { key.hasPrefix($0) || key.contains("." + $0) }
}

/// Holds encoders under their model keys while they load, so checkpoint paths apply unchanged.
private final class MediaEncoderStage: Module {
    private(set) var encoders: [String: Module]

    init(_ encoders: [String: Module]) {
        self.encoders = encoders
        super.init()
    }

    override func items() -> ModuleItems {
        ModuleItems(values: encoders.mapValues { .value(.module($0)) })
    }

    // Quantization replaces a top-level encoder, such as a projection `Linear`, through here.
    override func updateModule(key: String, _ value: Any) throws {
        guard encoders[key] != nil, let module = value as? Module else {
            throw UpdateError.unableToSet("media encoder \(key)")
        }
        encoders[key] = module
    }
}
