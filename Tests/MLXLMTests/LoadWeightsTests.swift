// Copyright © 2026 Apple Inc.

import Foundation
import MLX
import MLXNN
import XCTest

@testable import MLXLMCommon

/// A model whose head ships in a sidecar file that the index does not name, like
/// `jinaai/jina-reranker-v3-mlx` and its `projector.safetensors`.
private final class SidecarHeadModel: Module, BaseLanguageModel {
    @ModuleInfo(key: "layer") var layer: Linear
    @ModuleInfo(key: "projector") var projector: Linear

    let declaresSidecar: Bool

    init(declaresSidecar: Bool) {
        self.declaresSidecar = declaresSidecar
        _layer.wrappedValue = Linear(2, 2, bias: false)
        _projector.wrappedValue = Linear(2, 2, bias: false)
    }

    var additionalWeightFiles: [String] {
        declaresSidecar ? ["projector.safetensors"] : []
    }
}

final class LoadWeightsTests: XCTestCase {

    func testLoadWeightsUsesSafetensorsIndexWeightMapWhenPresent() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeEmptyFile("optiq_vision.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testLoadWeightsAppendsAdditionalFilesTheIndexOmits() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("projector.safetensors", in: directory)
        try writeEmptyFile("mtp.safetensors", in: directory)
        try writeIndex(["model.norm.weight": "model.safetensors"], in: directory)

        let names = try safetensorWeightURLs(
            in: directory, additionalFiles: ["projector.safetensors"]
        ).map(\.lastPathComponent)

        // the indexed shard is loaded first so its metadata wins, and the sidecar the index
        // omits is still loaded -- unrelated sidecars stay excluded
        XCTAssertEqual(names, ["model.safetensors", "projector.safetensors"])
    }

    func testLoadWeightsIgnoresAdditionalFilesThatAreAbsentOrAlreadyIndexed() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model.safetensors",
                "linear1.weight": "projector.safetensors",
            ], in: directory)
        try writeEmptyFile("projector.safetensors", in: directory)

        let names = try safetensorWeightURLs(
            in: directory, additionalFiles: ["projector.safetensors", "missing.safetensors"]
        ).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors", "projector.safetensors"])
    }

    func testLoadWeightsFallsBackToPresentFilesWhenTheIndexIsStale() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        // a quantized upload that kept the index of the unquantized source repo: it names
        // shards the repo does not ship
        try writeEmptyFile("model.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors"])
    }

    func testLoadWeightsFallsBackToPresentFilesWhenTheIndexIsPartiallyStale() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model-00001-of-00002.safetensors", in: directory)
        try writeIndex(
            [
                "model.norm.weight": "model-00001-of-00002.safetensors",
                "model.embed_tokens.weight": "model-00002-of-00002.safetensors",
            ], in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model-00001-of-00002.safetensors"])
    }

    func testLoadWeightsLoadsEveryFileWhenThereIsNoIndex() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeEmptyFile("model.safetensors", in: directory)
        try writeEmptyFile("projector.safetensors", in: directory)

        let names = try safetensorWeightURLs(in: directory).map(\.lastPathComponent)

        XCTAssertEqual(names, ["model.safetensors", "projector.safetensors"])
    }

    // MARK: - loadWeights end to end

    func testLoadWeightsReadsSidecarWeightsDeclaredByTheModel() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        let model = SidecarHeadModel(declaresSidecar: true)
        try loadWeights(modelDirectory: directory, model: model)

        XCTAssertEqual(model.projector.weight.asArray(Float.self), [1, 2, 3, 4])
    }

    func testLoadWeightsFailsWhenTheSidecarIsNotDeclared() throws {
        let directory = try makeTemporaryDirectory()
        defer { try? FileManager.default.removeItem(at: directory) }

        try writeSidecarCheckpoint(in: directory)

        // Without the declaration the index hides the sidecar and the head is never loaded --
        // `verify: [.all]` is what turns that into the keyNotFound error seen in #560.
        let model = SidecarHeadModel(declaresSidecar: false)
        XCTAssertThrowsError(try loadWeights(modelDirectory: directory, model: model))
    }

    /// Writes a checkpoint whose index names only `model.safetensors` while the head lives in
    /// `projector.safetensors`.
    private func writeSidecarCheckpoint(in directory: URL) throws {
        try save(
            arrays: ["layer.weight": MLXArray.zeros([2, 2])],
            url: directory.appendingPathComponent("model.safetensors"))
        try save(
            arrays: [
                "projector.weight": MLXArray(converting: [1.0, 2.0, 3.0, 4.0]).reshaped(2, 2)
            ],
            url: directory.appendingPathComponent("projector.safetensors"))
        try writeIndex(["layer.weight": "model.safetensors"], in: directory)
    }

    private func makeTemporaryDirectory() throws -> URL {
        let url = FileManager.default.temporaryDirectory
            .appendingPathComponent("LoadWeightsTests-\(UUID().uuidString)", isDirectory: true)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }

    private func writeEmptyFile(_ name: String, in directory: URL) throws {
        try Data().write(to: directory.appendingPathComponent(name))
    }

    private func writeIndex(_ weightMap: [String: String], in directory: URL) throws {
        let index: [String: Any] = [
            "metadata": ["total_size": 1],
            "weight_map": weightMap,
        ]
        let data = try JSONSerialization.data(withJSONObject: index)
        try data.write(to: directory.appendingPathComponent("model.safetensors.index.json"))
    }
}
