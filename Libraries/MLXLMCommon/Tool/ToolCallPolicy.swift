// Copyright © 2026 Apple Inc.

/// Rules for recovering and validating generated tool calls.
/// Native parser requirements and declared-tool authorization always apply.
public struct ToolCallPolicy: Hashable, Sendable {
    /// Bounded recovery of calls emitted in a different dialect.
    /// Applies to text formats; framed token protocols use their native parser.
    public var recovery: ToolCallRecoveryPolicy

    /// Schema enforcement after unambiguous argument normalization.
    public var validation: ToolCallValidationPolicy

    public init(
        recovery: ToolCallRecoveryPolicy = .conservative,
        validation: ToolCallValidationPolicy = .strict
    ) {
        self.recovery = recovery
        self.validation = validation
    }
}

/// Controls schema enforcement independently of syntax recovery and tool-name
/// authorization. Both modes normalize unambiguous, schema-declared values.
public enum ToolCallValidationPolicy: String, Hashable, Sendable, CaseIterable {
    /// Reject proven schema violations. Unsupported schema assertions remain
    /// unknown. This is the default, including for automatic tool dispatch.
    case strict
    /// Forward parsed arguments after normalization, even if they violate the
    /// schema, for applications that validate or repair arguments themselves.
    /// Native parser requirements and declared-tool authorization still apply.
    case permissive
}
