// Copyright © 2026 Apple Inc.

private enum OnyxRecipient: Equatable {
    case reasoning
    case user
    case tool(String)
    case unknown
}

private struct OnyxHeader: Equatable {
    var recipient: OnyxRecipient
}

private enum OnyxParseStep {
    case consumed
    case opened(OnyxHeader)
    case payload(OnyxHeader, Int)
    case closed(OnyxHeader, [Int])
}

/// Token-aware frame parser for the official Onyx control vocabulary.
///
/// Structural markers are compared as integer IDs. Literal marker-shaped text
/// in a payload can therefore never be mistaken for a frame boundary, and the
/// hot path avoids repeated whole-buffer string searches.
private struct OnyxFrameParser {
    private struct Controls {
        var start: Int
        var message: Int
        var eom: Int
        var eot: Int
    }

    private enum State {
        /// The initial `<|start|>assistant` lives in the generation prompt, so
        /// the first generated frame begins midway through its header.
        case header(tokens: [Int], primed: Bool)
        case payload(header: OnyxHeader, tokens: [Int])
        case betweenFrames
    }

    private let tokenizer: any Tokenizer
    private let controls: Controls
    private var state: State = .header(tokens: [], primed: true)
    private let maximumHeaderTokens = 64

    init?(tokenizer: any Tokenizer) {
        guard let start = tokenizer.convertTokenToId("<|start|>"),
            let message = tokenizer.convertTokenToId("<|message|>"),
            let eom = tokenizer.convertTokenToId("<|eom|>"),
            let eot = tokenizer.convertTokenToId("<|eot|>"),
            Set([start, message, eom, eot]).count == 4
        else { return nil }
        self.tokenizer = tokenizer
        self.controls = Controls(start: start, message: message, eom: eom, eot: eot)
    }

    mutating func push(_ token: Int) -> OnyxParseStep {
        if token == controls.start {
            state = .header(tokens: [], primed: false)
            return .consumed
        }

        switch state {
        case .header(var tokens, let primed):
            if token == controls.message {
                guard let header = parseHeader(tokens, primed: primed) else {
                    state = .betweenFrames
                    return .consumed
                }
                state = .payload(header: header, tokens: [])
                return .opened(header)
            }
            guard token != controls.eom, token != controls.eot,
                tokens.count < maximumHeaderTokens
            else {
                state = .betweenFrames
                return .consumed
            }
            tokens.append(token)
            state = .header(tokens: tokens, primed: primed)
            return .consumed

        case .payload(let header, var tokens):
            if token == controls.eom || token == controls.eot {
                state = .betweenFrames
                return .closed(header, tokens)
            }
            if case .tool = header.recipient {
                tokens.append(token)
            }
            state = .payload(header: header, tokens: tokens)
            return .payload(header, token)

        case .betweenFrames:
            return .consumed
        }
    }

    mutating func finish() -> OnyxParseStep? {
        defer { state = .betweenFrames }
        guard case .payload(let header, let tokens) = state else { return nil }
        return .closed(header, tokens)
    }

    private func parseHeader(_ tokens: [Int], primed: Bool) -> OnyxHeader? {
        let decoded = tokenizer.decode(tokenIds: tokens, skipSpecialTokens: false)
        let complete = (primed ? "assistant" : "") + decoded
        let fields = complete.split(whereSeparator: \.isWhitespace)
        guard fields.first == "assistant" else { return nil }
        guard let recipientField = fields.dropFirst().first(where: { $0.hasPrefix("to=") }) else {
            return OnyxHeader(recipient: .user)
        }
        let recipient = String(recipientField.dropFirst(3))
        switch recipient {
        case "self": return OnyxHeader(recipient: .reasoning)
        case "user": return OnyxHeader(recipient: .user)
        case "": return OnyxHeader(recipient: .unknown)
        default: return OnyxHeader(recipient: .tool(recipient))
        }
    }
}

/// Private protocol engine for the Onyx adapter. Generic consumers only see
/// `TokenStreamDecoder` and its protocol-neutral semantic events.
private struct OnyxProtocolDecoder {
    private var parser: OnyxFrameParser
    private let tokenizer: any Tokenizer
    private let toolParser = ATEMToolCallParser()
    private let tools: [[String: any Sendable]]?
    private var reasoningDetokenizer: NaiveStreamingDetokenizer
    private var responseDetokenizer: NaiveStreamingDetokenizer
    private(set) var isInsideReasoning = false

    init?(tokenizer: any Tokenizer, tools: [[String: any Sendable]]?) {
        guard let parser = OnyxFrameParser(tokenizer: tokenizer) else { return nil }
        self.parser = parser
        self.tokenizer = tokenizer
        self.tools = tools
        self.reasoningDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
        self.responseDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
    }

    mutating func push(_ token: Int) -> TokenStreamEvent? {
        route(parser.push(token))
    }

    mutating func finish() -> TokenStreamEvent? {
        parser.finish().flatMap { route($0) }
    }

    private mutating func route(_ step: OnyxParseStep) -> TokenStreamEvent? {
        switch step {
        case .consumed:
            isInsideReasoning = false
        case .opened(let header):
            isInsideReasoning = header.recipient == .reasoning
        case .payload(let header, let token):
            switch header.recipient {
            case .reasoning:
                isInsideReasoning = true
                reasoningDetokenizer.append(token: token)
                if let text = reasoningDetokenizer.next(), !text.isEmpty {
                    return .reasoning(text)
                }
            case .user:
                isInsideReasoning = false
                responseDetokenizer.append(token: token)
                if let text = responseDetokenizer.next(), !text.isEmpty {
                    return .response(text)
                }
            case .tool, .unknown:
                isInsideReasoning = false
            }

        case .closed(let header, let payload):
            isInsideReasoning = false
            switch header.recipient {
            case .reasoning:
                reasoningDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            case .user:
                responseDetokenizer = NaiveStreamingDetokenizer(tokenizer: tokenizer)
            case .tool(let recipient):
                let text = tokenizer.decode(tokenIds: payload, skipSpecialTokens: false)
                guard let call = toolParser.parse(content: text, tools: tools),
                    call.function.name == recipient
                else { return nil }
                return .toolCall(call)
            case .unknown:
                break
            }
        }
        return nil
    }
}

/// Token-stream decoder for Muse Glimmer's Onyx framing and ATEM tool calls.
struct OnyxStreamAdapter: TokenStreamDecoder {
    private var protocolDecoder: OnyxProtocolDecoder
    private var stopStringFilter: StopStringFilter
    let receivesStopTokens = true

    var isInsideReasoning: Bool { protocolDecoder.isInsideReasoning }

    init?(
        tokenizer: any Tokenizer,
        tools: [[String: any Sendable]]?,
        stopStrings: Set<String>
    ) {
        guard let decoder = OnyxProtocolDecoder(tokenizer: tokenizer, tools: tools) else {
            return nil
        }
        self.protocolDecoder = decoder
        self.stopStringFilter = StopStringFilter(stopStrings: stopStrings)
    }

    mutating func push(_ token: Int, emit: (TokenStreamEvent) -> Bool) -> Bool {
        guard let event = protocolDecoder.push(token) else { return true }
        return emitEvent(event, emit: emit)
    }

    mutating func finish(emit: (TokenStreamEvent) -> Bool) -> Bool {
        if let event = protocolDecoder.finish(), !emitEvent(event, emit: emit) { return false }
        if let text = stopStringFilter.finish() { return emit(.response(text)) }
        return true
    }

    private mutating func emitEvent(
        _ event: TokenStreamEvent,
        emit: (TokenStreamEvent) -> Bool
    ) -> Bool {
        switch event {
        case .reasoning(let text):
            return emit(.reasoning(text))
        case .response(let text):
            let result = stopStringFilter.process(text)
            if let response = result.text, !emit(.response(response)) { return false }
            if result.stopped {
                _ = emit(.stop)
                return false
            }
            return true
        case .toolCall(let call):
            if let response = stopStringFilter.finish(), !emit(.response(response)) {
                return false
            }
            return emit(.toolCall(call))
        case .stop:
            return emit(.stop)
        }
    }
}
