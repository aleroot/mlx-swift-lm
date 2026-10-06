// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

private final class Connector: Module {
    @ModuleInfo var gate: Linear
    @ModuleInfo var projection: Linear?

    override init() {
        _gate.wrappedValue = Linear(64, 64, bias: false)
        _projection.wrappedValue = Linear(64, 64, bias: false)
    }
}

/// A text model with a vision tower, and a vision projection nested in a module it shares.
private final class Captioner: Module, BaseLanguageModel, ExcludableComponentsProviding {
    @ModuleInfo(key: "language_model") var language: Linear
    @ModuleInfo(key: "vision_tower") var tower: Linear?
    @ModuleInfo var connector: Connector

    override init() {
        _language.wrappedValue = Linear(64, 64, bias: false)
        _tower.wrappedValue = Linear(64, 64, bias: false)
        _connector.wrappedValue = Connector()
    }

    var excludableComponents: [ModelComponent: [CheckpointComponent]] {
        [
            .vision: [
                CheckpointComponent(
                    name: "tower", namespaces: ["model.vision_tower"], destination: "vision_tower"),
                CheckpointComponent(
                    name: "projection", namespaces: ["model.connector.projection"],
                    destination: "connector.projection"),
            ]
        ]
    }

    /// Names of the tensors the loader passed to ``prepareCheckpoint(_:)``.
    var loadedNames = Set<String>()

    func prepareCheckpoint(_ checkpoint: ModelCheckpoint) throws -> ModelCheckpoint {
        loadedNames = Set(checkpoint.weights.keys)
        var checkpoint = checkpoint
        checkpoint.weights = dropModelScope(checkpoint.weights)
        return checkpoint
    }
}

/// The same layout, without support for excluding components.
private final class FixedCaptioner: Module, BaseLanguageModel {
    @ModuleInfo(key: "language_model") var language = Linear(64, 64, bias: false)
    @ModuleInfo(key: "vision_tower") var tower: Linear? = Linear(64, 64, bias: false)
    @ModuleInfo var connector = Connector()

    func sanitize(weights: [String: MLXArray]) -> [String: MLXArray] {
        dropModelScope(weights)
    }
}

/// Raw checkpoints scope everything under `model.`, as `transformers` exports do.
private func dropModelScope(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    Dictionary(
        uniqueKeysWithValues: weights.map { (String($0.dropFirst("model.".count)), $1) })
}

/// Declares a component whose module is not optional.
private final class MisdeclaredCaptioner: Module, BaseLanguageModel, ExcludableComponentsProviding {
    @ModuleInfo(key: "language_model") var language = Linear(64, 64, bias: false)

    var excludableComponents: [ModelComponent: [CheckpointComponent]] {
        [.vision: [.init(name: "language", namespaces: [], destination: "language_model")]]
    }
}

final class ModelComponentTests: XCTestCase {

    private let quantization = BaseConfiguration.Quantization(groupSize: 64, bits: 4)
    private let visionNamespaces = ["model.vision_tower", "model.connector.projection"]

    func testExcludedComponentLoadsWithoutItsTensors() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            excludedComponents: [.vision])

        XCTAssertEqual(model.loadedNames, Set(raw(checkpoint.text).keys))
        XCTAssertNil(model.tower)
        XCTAssertNil(model.connector.projection)
        XCTAssertTrue(model.language is QuantizedLinear)
        try assertParameters(of: model, match: checkpoint.text)
    }

    func testEveryComponentLoadsByDefault() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization)

        XCTAssertNotNil(model.tower)
        XCTAssertTrue(model.connector.projection is QuantizedLinear)
        try assertParameters(
            of: model, match: checkpoint.text.merging(checkpoint.visual) { a, _ in a })
    }

    func testComponentsAModelCannotExcludeStillLoad() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let fixed = FixedCaptioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: fixed, quantization: quantization,
            excludedComponents: [.vision])
        XCTAssertNotNil(fixed.tower)

        let model = Captioner()
        try loadWeights(
            modelDirectory: checkpoint.directory, model: model, quantization: quantization,
            excludedComponents: [ModelComponent("audio")])
        XCTAssertNotNil(model.tower)
    }

    func testComponentDestinationMustBeAnOptionalModule() {
        XCTAssertThrowsError(
            try removeExcludedComponents([.vision], from: MisdeclaredCaptioner())
        ) { error in
            XCTAssertTrue(error is UpdateError, "\(error)")
        }
    }

    func testExcludedNamespacesKeepTheMetadataOfSkippedFiles() throws {
        let checkpoint = try writeCheckpoint()
        defer { try? FileManager.default.removeItem(at: checkpoint.directory) }

        let loaded = try loadModelCheckpoint(
            urls: [checkpoint.textFile, checkpoint.vision], excludedNamespaces: visionNamespaces)

        XCTAssertEqual(Set(loaded.weights.keys), Set(raw(checkpoint.text).keys))
        XCTAssertEqual(loaded.metadata, ["source": "vision shard"])
    }

    func testExcludedNamespacesMatchWholePathComponents() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }
        let url = directory.appendingPathComponent("model.safetensors")
        try save(
            arrays: [
                "model.vision_tower.weight": MLXArray.zeros([2]),
                "model.vision_tower_norm.weight": MLXArray.zeros([2]),
            ], url: url)

        let loaded = try loadModelCheckpoint(
            urls: [url], excludedNamespaces: ["model.vision_tower"])

        XCTAssertEqual(Set(loaded.weights.keys), ["model.vision_tower_norm.weight"])
    }

    // MARK: - Fixtures

    private struct Checkpoint {
        let directory: URL
        let textFile: URL
        let vision: URL
        let text: [String: MLXArray]
        let visual: [String: MLXArray]
    }

    /// Writes a raw checkpoint in two shards: text weights, then vision weights.
    ///
    /// The language model and the vision projection are quantized; the tower is not.
    private func writeCheckpoint() throws -> Checkpoint {
        let directory = try makeTemporaryDirectory()
        let source = Captioner()
        quantize(model: source, groupSize: 64, bits: 4) { path, _ in
            path == "language_model" || path == "connector.projection"
        }
        var text = [String: MLXArray]()
        var visual = [String: MLXArray]()
        for (key, value) in source.parameters().flattened() {
            if key.hasPrefix("vision_tower") || key.hasPrefix("connector.projection") {
                visual[key] = value
            } else {
                text[key] = value
            }
        }
        let textFile = directory.appendingPathComponent("model-00001-of-00002.safetensors")
        let vision = directory.appendingPathComponent("model-00002-of-00002.safetensors")
        try save(arrays: raw(text), url: textFile)
        try save(arrays: raw(visual), metadata: ["source": "vision shard"], url: vision)
        return Checkpoint(
            directory: directory, textFile: textFile, vision: vision, text: text, visual: visual)
    }

    private func raw(_ weights: [String: MLXArray]) -> [String: MLXArray] {
        Dictionary(uniqueKeysWithValues: weights.map { ("model.\($0)", $1) })
    }

    private func assertParameters(of model: Module, match expected: [String: MLXArray]) throws {
        let actual = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
        XCTAssertEqual(Set(actual.keys), Set(expected.keys))
        for (key, value) in expected {
            XCTAssertTrue(arrayEqual(try XCTUnwrap(actual[key]), value).item(Bool.self), key)
        }
    }

    private func makeTemporaryDirectory() throws -> URL {
        let directory = FileManager.default.temporaryDirectory
            .appendingPathComponent("ModelComponentTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return directory
    }
}
