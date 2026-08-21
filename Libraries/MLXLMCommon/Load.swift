// Copyright © 2024 Apple Inc.

import Foundation
import MLX
import MLXNN

private struct SafetensorsIndex: Decodable {
    let weightMap: [String: String]

    enum CodingKeys: String, CodingKey {
        case weightMap = "weight_map"
    }
}

/// The `safetensors` files in `modelDirectory` that hold the model's weights.
///
/// When `model.safetensors.index.json` is present it is authoritative: only the files it names
/// are loaded.  That keeps sidecar checkpoints that belong to a different module -- for example
/// `mtp.safetensors` or `optiq_vision.safetensors` -- out of the main model.  Two cases relax
/// that rule:
///
/// - The index names files the repository does not ship.  Some quantized uploads carry the
///   index over from the unquantized source repo without regenerating it, so it names shards
///   that do not exist.  Loading only those files would load nothing, so every `safetensors`
///   file that is actually present is used instead.
/// - `additionalFiles` names sidecars the model requires but the index omits, for example the
///   Jina reranker's `projector.safetensors`.  Missing entries are ignored.
///
/// - Parameters:
///   - modelDirectory: directory holding the weight files
///   - additionalFiles: file names, relative to `modelDirectory`, to load in addition to the
///     files named by the index.  See ``BaseLanguageModel/additionalWeightFiles``.
package func safetensorWeightURLs(
    in modelDirectory: URL, additionalFiles: [String] = []
) throws -> [URL] {
    let indexURL = modelDirectory.appendingPathComponent("model.safetensors.index.json")
    guard FileManager.default.fileExists(atPath: indexURL.path) else {
        return allSafetensorWeightURLs(in: modelDirectory)
    }

    let data = try Data(contentsOf: indexURL)
    let index = try JSONDecoder().decode(SafetensorsIndex.self, from: data)
    let indexed = Set(index.weightMap.values)
        .sorted()
        .map { modelDirectory.appendingPathComponent($0) }

    // an index that names files the repo does not ship is stale -- load what is there
    guard !indexed.isEmpty,
        indexed.allSatisfy({ FileManager.default.fileExists(atPath: $0.path) })
    else {
        return allSafetensorWeightURLs(in: modelDirectory)
    }

    var seen = Set(indexed.map(\.standardizedFileURL.path))
    var urls = indexed
    for name in additionalFiles {
        let url = modelDirectory.appendingPathComponent(name)
        guard FileManager.default.fileExists(atPath: url.path),
            seen.insert(url.standardizedFileURL.path).inserted
        else {
            continue
        }
        urls.append(url)
    }
    return urls
}

private func allSafetensorWeightURLs(in modelDirectory: URL) -> [URL] {
    let enumerator = FileManager.default.enumerator(
        at: modelDirectory, includingPropertiesForKeys: nil)!
    return
        enumerator
        .compactMap { item -> URL? in
            guard let url = item as? URL, url.pathExtension == "safetensors" else {
                return nil
            }
            return url
        }
        .sorted { $0.path < $1.path }
}

/// Load model weights.
///
/// This is typically called via ``GenericModelFactory/load(from:using:configuration:useLatest:progressHandler:)``.
/// This function loads model weight `safetensor` files in the given `modelDirectory`,
/// calls ``BaseLanguageModel/sanitize(weights:metadata:)`` to allow per-model preprocessing,
/// applies optional quantization, and
/// updates the model with the weights.
public func loadWeights(
    modelDirectory: URL, model: BaseLanguageModel,
    quantization: BaseConfiguration.Quantization? = nil,
    perLayerQuantization: BaseConfiguration.PerLayerQuantization? = nil
) throws {
    // load the weights and collect metadata from the first safetensor file
    var weights = [String: MLXArray]()
    var metadata = [String: String]()
    for url in try safetensorWeightURLs(
        in: modelDirectory, additionalFiles: model.additionalWeightFiles)
    {
        let (w, m) = try loadArraysAndMetadata(url: url)
        for (key, value) in w {
            weights[key] = value
        }
        if metadata.isEmpty {
            metadata = m
        }
    }

    // per-model cleanup (models can inspect metadata to customize behavior)
    weights = model.sanitize(weights: weights, metadata: metadata)

    // quantize if needed
    if quantization != nil || perLayerQuantization != nil {
        quantize(model: model) { path, module in
            if weights["\(path).scales"] != nil {
                if let perLayerQuantization {
                    return perLayerQuantization.quantization(layer: path)?.asTuple
                } else {
                    return quantization?.asTuple
                }
            } else {
                return nil
            }
        }
    }

    // apply the loaded weights
    let parameters = ModuleParameters.unflattened(weights)
    try model.update(parameters: parameters, verify: [.all])

    eval(model)
}
