// Copyright © 2026 Apple Inc.

import Foundation
import XCTest
@testable import MLXLLM

public final class DeepseekV2Tests: XCTestCase {

    private func configData() -> Data {
        """
        {
            "model_type": "deepseek_v2",
            "vocab_size": 1024,
            "hidden_size": 64,
            "intermediate_size": 128,
            "moe_intermediate_size": 32,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "n_shared_experts": 1,
            "n_routed_experts": 2,
            "routed_scaling_factor": 1.0,
            "kv_lora_rank": 8,
            "q_lora_rank": null,
            "qk_rope_head_dim": 8,
            "v_head_dim": 8,
            "qk_nope_head_dim": 8,
            "topk_method": "group_limited_greedy",
            "n_group": 2,
            "topk_group": 1,
            "num_experts_per_tok": 1,
            "moe_layer_freq": 1,
            "first_k_dense_replace": 0,
            "max_position_embeddings": 4096,
            "rms_norm_eps": 1e-6,
            "rope_theta": 10000.0,
            "rope_scaling": {
                "type": "yarn",
                "factor": 4,
                "original_max_position_embeddings": 1024,
                "beta_fast": 32,
                "beta_slow": 1,
                "mscale": 0.707,
                "mscale_all_dim": 0.707
            },
            "attention_bias": false
        }
        """.data(using: .utf8)!
    }

    func testConfigurationDecodesNullQLoraRank() throws {
        let config = try JSONDecoder().decode(
            DeepseekV2Configuration.self, from: configData())

        XCTAssertNil(config.qLoraRank)
        XCTAssertEqual(config.topkMethod, "group_limited_greedy")
        XCTAssertEqual(config.nGroup, 2)
        XCTAssertEqual(config.topkGroup, 1)
    }

    func testConfigurationDecodesRopeScaling() throws {
        let config = try JSONDecoder().decode(
            DeepseekV2Configuration.self, from: configData())

        XCTAssertEqual(config.ropeScaling?["type"], .string("yarn"))
        XCTAssertEqual(config.ropeScaling?["factor"], .int(4))
        XCTAssertEqual(config.ropeScaling?["original_max_position_embeddings"], .int(1024))
    }
}
